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

## Supported Tasks

- Gait pattern classification (normal / abnormal)
- Walking phase detection (stance, swing, double support)
- Gait parameter estimation (cadence, stride length, symmetry)
