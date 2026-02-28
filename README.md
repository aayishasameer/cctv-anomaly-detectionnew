# CCTV Anomaly Detection System

Real-time CCTV surveillance system with person detection, ReID tracking, and anomaly/theft detection.

## Features

- **YOLOv8 + BoT-SORT** – Person detection and tracking
- **OSNet ReID** – Cross-camera person re-identification
- **VAE Anomaly Detection** – Behavioral anomaly detection
- **MediaPipe** – Hand and pose estimation (when available)
- **Stealing Detection** – Multi-level theft detection with zone learning

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train VAE model (required, use normal behavior videos)
python train_vae_model.py

# Run complete system
python complete_cctv_system.py -i your_video.mp4
```

## Main Scripts

- `complete_cctv_system.py` – Full pipeline with ReID and anomaly visualization
- `stealing_detection_system.py` – Theft detection with hand-zone interaction
- `adaptive_zone_learning.py` – Learn interaction zones from normal behavior
- `train_vae_model.py` – Train VAE on normal behavior features

## License

MIT
