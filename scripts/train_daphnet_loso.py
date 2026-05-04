"""Leave-One-Subject-Out (LOSO) cross-validation on Daphnet FOG dataset.

Each of the 10 subjects is held out as test once; the remaining 9 train the model.
This avoids the data leakage in random_split and gives an honest generalisation estimate.
"""

import time
import pathlib
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.daphnet_dataset import DaphnetDataset, CLASS_NAMES, load_by_subject
from src.models.multimodal_gait_net import IMUGaitNet
from src.utils.metrics import compute_metrics


def _make_dataset(imu_wins, labels, cfg, rng_seed):
    return DaphnetDataset(
        window=cfg["data"]["window"],
        grid_size=tuple(cfg["data"]["pressure_grid_size"]),
        num_joints=cfg["data"]["skeleton_joints"],
        rng_seed=rng_seed,
        _preloaded=(imu_wins, labels),
    )


def _train_fold(
    train_ds, val_ds, test_ds, config, device, out_path, fold_id
):
    tcfg = config["training"]

    labels_arr = np.array(train_ds.labels)
    n_pos = (labels_arr == 1).sum()
    n_neg = (labels_arr == 0).sum()
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg["batch_size"], num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=tcfg["batch_size"], num_workers=0)

    model = IMUGaitNet(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, tcfg["epochs"] - tcfg["scheduler"]["warmup_epochs"])
    )

    best_val_f1 = 0.0
    patience_counter = 0
    es = tcfg["early_stopping"]

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()

        # Train
        model.train()
        all_preds, all_lbls, total_loss = [], [], 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lbl = batch.pop("label")
            logits = model(batch)
            loss = criterion(logits, lbl)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * lbl.size(0)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_lbls.append(lbl.cpu().numpy())

        tm = compute_metrics(np.concatenate(all_lbls), np.concatenate(all_preds))
        tm["loss"] = total_loss / sum(len(p) for p in all_preds)

        # Val
        model.eval()
        all_preds, all_lbls, total_loss = [], [], 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                lbl = batch.pop("label")
                logits = model(batch)
                total_loss += criterion(logits, lbl).item() * lbl.size(0)
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_lbls.append(lbl.cpu().numpy())

        vm = compute_metrics(np.concatenate(all_lbls), np.concatenate(all_preds))
        vm["loss"] = total_loss / sum(len(p) for p in all_preds)

        if epoch > tcfg["scheduler"]["warmup_epochs"]:
            scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Ep {epoch:2d}/{tcfg['epochs']} | "
            f"Train Acc {tm['accuracy']:.3f} | "
            f"Val Acc {vm['accuracy']:.3f} F1 {vm['f1_macro']:.3f} | "
            f"{elapsed:.1f}s"
        )

        if vm["f1_macro"] > best_val_f1 + es["min_delta"]:
            best_val_f1 = vm["f1_macro"]
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "config": config}, out_path)
        else:
            patience_counter += 1
            if patience_counter >= es["patience"]:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Test
    ckpt = torch.load(out_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lbl = batch.pop("label")
            all_preds.append(model(batch).argmax(1).cpu().numpy())
            all_lbls.append(lbl.cpu().numpy())

    return compute_metrics(np.concatenate(all_lbls), np.concatenate(all_preds))


def run(config_path="configs/daphnet.yaml", output_dir="outputs/daphnet_loso"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = config["data"]
    print("Loading all subjects...", flush=True)
    subjects = load_by_subject(
        data_cfg["data_dir"],
        window=data_cfg["window"],
        stride=data_cfg["stride"],
    )
    subject_ids = sorted(subjects.keys())
    print(f"Subjects: {subject_ids}\n")

    fold_results = []

    for fold_idx, test_subj in enumerate(subject_ids):
        print("=" * 70)
        print(f"Fold {fold_idx+1}/{len(subject_ids)}  |  Test subject: {test_subj}")
        print("=" * 70)

        # Gather train subjects (all except test)
        train_imu, train_lbl = [], []
        for s in subject_ids:
            if s == test_subj:
                continue
            train_imu.extend(subjects[s][0])
            train_lbl.extend(subjects[s][1])

        test_imu, test_lbl = subjects[test_subj]

        # Use 15% of train as in-fold validation
        n_val = max(1, int(len(train_imu) * 0.15))
        rng = np.random.default_rng(42 + fold_idx)
        idx = rng.permutation(len(train_imu))
        val_idx, tr_idx = idx[:n_val], idx[n_val:]

        val_imu   = [train_imu[i] for i in val_idx]
        val_lbl   = [train_lbl[i] for i in val_idx]
        tr_imu    = [train_imu[i] for i in tr_idx]
        tr_lbl    = [train_lbl[i] for i in tr_idx]

        train_ds = _make_dataset(tr_imu,   tr_lbl,   config, rng_seed=42+fold_idx)
        val_ds   = _make_dataset(val_imu,  val_lbl,  config, rng_seed=42+fold_idx)
        test_ds  = _make_dataset(test_imu, test_lbl, config, rng_seed=42+fold_idx)

        n_pos = sum(test_lbl)
        print(f"  Train: {len(tr_imu):5d}  Val: {len(val_imu):4d}  Test: {len(test_imu):4d} "
              f"(freeze={n_pos}/{len(test_imu)})")

        metrics = _train_fold(
            train_ds, val_ds, test_ds, config, device,
            out / f"fold_{test_subj}.pt", test_subj,
        )
        fold_results.append((test_subj, metrics))

        print(f"\n  >> {test_subj} Test | "
              f"Acc {metrics['accuracy']:.4f}  "
              f"F1 {metrics['f1_macro']:.4f}  "
              f"Prec {metrics['precision']:.4f}  "
              f"Rec {metrics['recall']:.4f}")
        print(f"     Confusion:\n{metrics['confusion_matrix']}\n")

    # Aggregate
    print("\n" + "=" * 70)
    print("LOSO CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print(f"{'Subject':<10} {'Accuracy':>10} {'F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 54)
    accs, f1s, precs, recs = [], [], [], []
    for subj, m in fold_results:
        print(f"{subj:<10} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} "
              f"{m['precision']:>10.4f} {m['recall']:>10.4f}")
        accs.append(m['accuracy']); f1s.append(m['f1_macro'])
        precs.append(m['precision']); recs.append(m['recall'])
    print("-" * 54)
    print(f"{'Mean':<10} {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f} "
          f"{np.mean(precs):>10.4f} {np.mean(recs):>10.4f}")
    print(f"{'Std':<10} {np.std(accs):>10.4f} {np.std(f1s):>10.4f} "
          f"{np.std(precs):>10.4f} {np.std(recs):>10.4f}")

    np.save(out / "loso_results.npy", {s: m for s, m in fold_results})
    print(f"\n결과 저장: {out}/loso_results.npy")


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/daphnet.yaml"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/daphnet_loso"
    run(cfg, out)
