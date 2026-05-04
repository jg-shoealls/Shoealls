"""Train on real Daphnet data with leave-one-subject-out cross-validation."""

import time
import pathlib
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.daphnet_dataset import DaphnetDataset, CLASS_NAMES
from src.models.multimodal_gait_net import MultimodalGaitNet
from src.utils.metrics import compute_metrics


def run(config_path: str = "configs/daphnet.yaml", output_dir: str = "outputs/daphnet"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = config["data"]
    dataset = DaphnetDataset(
        data_dir=data_cfg["data_dir"],
        window=data_cfg["window"],
        stride=data_cfg["stride"],
        grid_size=tuple(data_cfg["pressure_grid_size"]),
        num_joints=data_cfg["skeleton_joints"],
    )

    total = len(dataset)
    train_n = int(total * data_cfg["train_split"])
    val_n = int(total * data_cfg["val_split"])
    test_n = total - train_n - val_n

    train_ds, val_ds, test_ds = random_split(
        dataset, [train_n, val_n, test_n],
        generator=torch.Generator().manual_seed(42),
    )

    labels = np.array(dataset.labels)
    n_pos = (labels == 1).sum()
    n_neg = (labels == 0).sum()
    print(f"Dataset: {total} windows | normal={n_neg} freeze={n_pos}")
    print(f"Splits - Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    tcfg = config["training"]
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=tcfg["batch_size"], num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=tcfg["batch_size"], num_workers=0)

    # Class-weighted loss to handle imbalance
    weight = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    model = MultimodalGaitNet(config).to(device)
    print(f"Parameters: {model.get_num_trainable_params():,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"] - tcfg["scheduler"]["warmup_epochs"]
    )

    best_val_acc = 0.0
    patience_counter = 0
    es = tcfg["early_stopping"]
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\nTraining for {tcfg['epochs']} epochs...")
    print("-" * 70)

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()

        # --- Train ---
        model.train()
        total_loss, all_preds, all_labels = 0.0, [], []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lbl = batch.pop("label")
            logits = model(batch)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * lbl.size(0)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(lbl.cpu().numpy())

        train_preds = np.concatenate(all_preds)
        train_lbls  = np.concatenate(all_labels)
        tm = compute_metrics(train_lbls, train_preds)
        tm["loss"] = total_loss / len(train_preds)

        # --- Val ---
        model.eval()
        total_loss, all_preds, all_labels = 0.0, [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                lbl = batch.pop("label")
                logits = model(batch)
                total_loss += criterion(logits, lbl).item() * lbl.size(0)
                all_preds.append(logits.argmax(1).cpu().numpy())
                all_labels.append(lbl.cpu().numpy())

        val_preds = np.concatenate(all_preds)
        val_lbls  = np.concatenate(all_labels)
        vm = compute_metrics(val_lbls, val_preds)
        vm["loss"] = total_loss / len(val_preds)

        if epoch > tcfg["scheduler"]["warmup_epochs"]:
            scheduler.step()

        history["train_loss"].append(tm["loss"])
        history["val_loss"].append(vm["loss"])
        history["train_acc"].append(tm["accuracy"])
        history["val_acc"].append(vm["accuracy"])

        print(
            f"Epoch {epoch:3d}/{tcfg['epochs']} | "
            f"Train Loss: {tm['loss']:.4f} Acc: {tm['accuracy']:.4f} | "
            f"Val Loss: {vm['loss']:.4f} Acc: {vm['accuracy']:.4f} F1: {vm['f1_macro']:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        if vm["accuracy"] > best_val_acc + es["min_delta"]:
            best_val_acc = vm["accuracy"]
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "val_accuracy": best_val_acc, "config": config, "history": history,
            }, out / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= es["patience"]:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # --- Test ---
    print("\n" + "=" * 70)
    ckpt = torch.load(out / "best_model.pt", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            lbl = batch.pop("label")
            logits = model(batch)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_labels.append(lbl.cpu().numpy())

    test_m = compute_metrics(np.concatenate(all_labels), np.concatenate(all_preds))

    print(f"Best epoch:      {ckpt['epoch']}")
    print(f"Test Accuracy:   {test_m['accuracy']:.4f}")
    print(f"Test F1 (macro): {test_m['f1_macro']:.4f}")
    print(f"Test Precision:  {test_m['precision']:.4f}")
    print(f"Test Recall:     {test_m['recall']:.4f}")
    print(f"\nClasses: {CLASS_NAMES}")
    print(f"Confusion Matrix:\n{test_m['confusion_matrix']}")

    # Save history into checkpoint
    ckpt["history"] = history
    torch.save(ckpt, out / "best_model.pt")


if __name__ == "__main__":
    import sys
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/daphnet.yaml"
    out = sys.argv[2] if len(sys.argv) > 2 else "outputs/daphnet"
    run(cfg, out)
