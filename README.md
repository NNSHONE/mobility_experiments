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

If `metrics.py` uses torch:
```bash
pip install torch torchvision matplotlib
```

### 2. Prepare Data
- Convert YOLO `.txt` GT files to `gt_annotations.json`
- Save `predictions.json` from YOLOv11
- Update paths in `evaluate.py`

### 3. Run Evaluation
```bash
python evaluate.py
```

### 4. Output
- Console: Precision, Recall, mAP@0.5, mAP@0.5:0.95, F1
- Saved files: `metrics_out/*.png` PR/F1/Confidence curves

---

## 📊 Evaluation Metrics
- `Precision`, `Recall`, `F1 Score`
- `mAP@0.5`, `mAP@0.5:0.95`
- Per-class AP (from DetMetrics)

## 🔐 Access Control
This repo is **Private**. Only invited collaborators can access.  
To share:
- Go to `Settings` → `Collaborators`
- Invite GitHub accounts manually

---

## 🙋‍♂️ Author & License
- Author: `your_name`
- License: MIT or custom (your choice)

---

## ✅ To Do
- [ ] Class-wise AP to CSV export
- [ ] COCO Eval integration (pycocotools)
- [ ] Jupyter version for visual debugging


