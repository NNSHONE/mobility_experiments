# Mobility Expreiments
## YOLOv11 train

This repository provides a full evaluation pipeline for YOLOv11 model predictions using custom ground truth and prediction files.

## Custom dataset
```
├── test       # Flat format ground truth (from YOLO .txts)├
│    │
│    ├──
├── train      # YOLOv11 inference output (COCO format)
│    │
│
├── valid      # ../train
│    │
│
├── data.yaml  # YOLOv11 yaml file

```

## 🚀 Usage
### 1. Install Requirements
```bash
pip install -r requirements.txt
```


