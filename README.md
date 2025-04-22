# YOLOv11 객체 검출 프로젝트

## 목차
* [소개](#소개)
* [설치](#설치)
* [사용법](#사용법)
* [성능 평가](#성능-평가)

## 소개
YOLOv11을 활용한 객체 검출 프로젝트입니다. 모델 학습, 실시간 웹캠 감지, 이미지 기반 객체 검출 기능을 제공합니다.

## 설치
```bash
# 필수 라이브러리 설치
pip install ultralytics
pip install opencv-python
pip install pandas
pip install numpy
```

## 사용법
### 1. 모델 학습
```bash
python yolov11.py --train --data [데이터셋 YAML 경로] --save_dir [저장 경로]
```

### 2. 웹캠 실시간 객체 검출
```bash
python yolov11.py --test_cam
```

### 3. 이미지 객체 검출
```bash
python yolov11.py --test_image [이미지 경로]
```

## 성능 평가
### 저장되는 메트릭
- mAP@0.5
- mAP@0.5:0.95
- Precision
- Recall
- F1 Score

### 결과 저장 형식
```
detection_metrics_results.xlsx
├── Class Metrics
│   ├── Class Name
│   ├── AP@0.5
│   ├── AP@0.5:0.95
│   ├── Precision
│   ├── Recall
│   └── F1-Score
└── Overall Metrics
    ├── mAP@0.5
    ├── mAP@0.5:0.95
    ├── mean_precision
    └── mean_recall
```

## 요구사항
- Python 3.6+
- CUDA 지원 GPU (선택)
- 웹캠 (실시간 테스트용)
