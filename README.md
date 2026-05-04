# Multimodal Walking AI Algorithm

멀티모달 보행 데이터 기반 AI 알고리즘 설계 및 초기 검증

## Overview

This project implements a multimodal AI algorithm for gait analysis, combining:
- **IMU (Inertial Measurement Unit)** sensor data (accelerometer, gyroscope)
- **Pressure sensor** data (foot plantar pressure distribution)
- **Skeleton joint** data (body keypoint positions)

## Architecture

```
IMU Data ──────► IMU Encoder (1D-CNN + LSTM) ──┐
                                                ├── Cross-Modal Attention Fusion ──► Classifier
Pressure Data ─► Pressure Encoder (2D-CNN) ────┤
                                                │
Skeleton Data ─► Skeleton Encoder (GCN) ───────┘
```

## Project Structure

```
src/
├── data/           # Data loading and preprocessing
├── models/         # Model architectures
├── training/       # Training pipeline
├── validation/     # Evaluation and validation
└── utils/          # Utility functions
configs/            # Configuration files
tests/              # Unit tests
```

## Quick Start

```bash
pip install -r requirements.txt
python -m src.training.train --config configs/default.yaml
python -m src.validation.validate --checkpoint outputs/best_model.pt
```

## WearGait-PD to Google Drive

To avoid storing WearGait-PD locally, stream each Synapse file through a temporary
directory and upload it to Google Drive:

```bash
python scripts/sync_weargait_to_gdrive.py ^
  --synapse-token <SYNAPSE_PAT> ^
  --service-account-json <google-service-account.json>
```

For a personal Drive folder, share the destination folder with the service
account email first. To use user OAuth instead:

```bash
python scripts/sync_weargait_to_gdrive.py ^
  --synapse-token <SYNAPSE_PAT> ^
  --oauth-client-json <oauth-client.json>
```

Useful checks:

```bash
python scripts/sync_weargait_to_gdrive.py --synapse-token <SYNAPSE_PAT> --oauth-client-json <oauth-client.json> --dry-run --max-files 5
```

By default, files are uploaded under a `WearGait-PD` folder in My Drive. Pass
`--drive-folder-id <GOOGLE_DRIVE_FOLDER_ID>` to use another parent folder.

## Supported Tasks

- Gait pattern classification (normal / abnormal)
- Walking phase detection (stance, swing, double support)
- Gait parameter estimation (cadence, stride length, symmetry)
