# SHADE: From Shadows to Highlights

**A Compact RAW Pipeline for Backlit Low-Light Image Enhancement**

> [Paper](#) | [Dataset](#dataset) | [Supplementary](#supplementary-material)

---

## Overview

SHADE is a lightweight RAW-domain pipeline designed to enhance backlit and low-light images directly from camera RAW data. By operating in the RAW domain before the traditional ISP (Image Signal Processing) chain, SHADE recovers shadow detail and preserves highlight integrity with minimal computational overhead.

Key features:
- Processes RAW (Bayer) sensor data for maximum dynamic-range recovery
- Compact architecture suitable for on-device and real-time deployment
- Handles challenging backlit scenes where conventional methods fail
- Paired with a purpose-built backlit low-light dataset

---

## News

- **2026-04-27** — Repository created; code, dataset, and supplementary material coming soon.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/xenbaloch/SHADE.git
cd SHADE

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Repository Structure

```
SHADE/
├── shade/                  # Core SHADE package
│   ├── __init__.py
│   ├── model.py            # Network architecture
│   ├── pipeline.py         # RAW-domain processing pipeline
│   └── utils.py            # Helper utilities (RAW I/O, metrics, etc.)
├── scripts/
│   ├── train.py            # Training script
│   └── inference.py        # Single-image / batch inference
├── requirements.txt
└── README.md
```

---

## Usage

### Inference on a single RAW image

```bash
python scripts/inference.py \
    --input  path/to/input.dng \
    --output path/to/output.png \
    --weights path/to/shade_weights.pth
```

### Training from scratch

```bash
python scripts/train.py \
    --data_root path/to/dataset \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4
```

---

## Dataset

The backlit low-light dataset used in this work will be released here.

> **Link:** _coming soon_

The dataset contains paired RAW / reference images captured under diverse backlit and low-light conditions. Download it and place the files under `data/` before running the training script.

---

## Supplementary Material

Supplementary figures, ablation results, and additional qualitative comparisons will be added here.

> **Link:** _coming soon_

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{shade2026,
  title   = {From Shadows to Highlights: A Compact RAW Pipeline for Backlit Low-Light Image Enhancement},
  author  = {Baloch, Xen and others},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
