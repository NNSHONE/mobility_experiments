import argparse
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np

def train_val_model(data_path,save_dir):
    model = YOLO('/workspace/seonghyeon/mobility/runs/detect/train/weights/best.pt') #학습 전 : 초기 모델 경로 // 학습 후 : 현재 학습 모델 경로
    # results_train = model.train(
    #     data=data_path,
    #     epochs=20, # 학습 횟수
    #     imgsz=640, # 학습 이미지 사이즈
    #     device=0 # 사용자의 컴퓨팅 자원에 맞춰 설정 (e.g., 'cpu', 0, [0,1,2,3])
    # )
    ###############################################################################
    results_val = model.val(
        data=data_path,
        split="test",  
        save_json=True, # 검증 결과 json 파일로 저장 True or False
        save_hybrid=False, 
        save=False,
        project=save_dir,
        name="eval",
        imgsz=640,
        batch=1, # 사용자의 컴퓨팅 자원에 맞춰 설정
        device="cpu", # 사용자의 컴퓨팅 자원에 맞춰 설정 ('cpu', 0, [0,1,2,3])
        plots=True
    )
    ###############################################################################
    metrics = {
        'ap50': results_val.box.ap50,  # AP@0.5
        'ap': results_val.box.ap,      # AP@0.5:0.95
        'precision': results_val.box.p,  # Precision
        'recall': results_val.box.r,     # Recall
        'f1': results_val.box.f1         # F1 Score
    }
    
    # 전체 모델 성능 메트릭
    overall_metrics = {
        'mAP@0.5': results_val.box.map50,
        'mAP@0.5:0.95': results_val.box.map,
        'mean_precision': results_val.box.mp,
        'mean_recall': results_val.box.mr
    }

    class_names = model.names  # 클래스 이름 매핑

    # 클래스별 메트릭 DataFrame 생성
    class_metrics = []
    for i in range(len(class_names)):
        class_metric = {
            'Class Name': class_names[i],
            'AP@0.5': metrics['ap50'][i] if isinstance(metrics['ap50'], (list, np.ndarray)) else metrics['ap50'],
            'AP@0.5:0.95': metrics['ap'][i] if isinstance(metrics['ap'], (list, np.ndarray)) else metrics['ap'],
            'Precision': metrics['precision'][i] if isinstance(metrics['precision'], (list, np.ndarray)) else metrics['precision'],
            'Recall': metrics['recall'][i] if isinstance(metrics['recall'], (list, np.ndarray)) else metrics['recall'],
            'F1-Score': metrics['f1'][i] if isinstance(metrics['f1'], (list, np.ndarray)) else metrics['f1']
        }
        class_metrics.append(class_metric)

    # DataFrame으로 변환
    df_class_metrics = pd.DataFrame(class_metrics)
    df_overall_metrics = pd.DataFrame([overall_metrics])

    # 결과 저장 (엑셀 파일)
    with pd.ExcelWriter('/workspace/seonghyeon/detection_metrics_results.xlsx') as writer:
        df_class_metrics.to_excel(writer, sheet_name='Class Metrics', index=False)
        df_overall_metrics.to_excel(writer, sheet_name='Overall Metrics', index=False)

    # 평가 결과 출력
    print("\n=== Overall Detection Metrics ===")
    for k, v in overall_metrics.items():
        print(f"{k}: {v:.4f}")

    return results_val

def test_model_webcam():
    model = YOLO('/workspace/seonghyeon/yolo11n.pt')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("[INFO] Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, imgsz=640, conf=0.3, stream=True)

        for r in results:
            annotated_frame = r.plot()
            cv2.imshow('YOLOv11 Webcam Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_image(image_path):
    model = YOLO('/workspace/seonghyeon/yolo11n.pt')
    results = model.predict(source=image_path, imgsz=640, conf=0.3)

    for r in results:
        annotated = r.plot()
        cv2.imshow("YOLOv11 Image Prediction", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def main():
    parser = argparse.ArgumentParser(description="YOLOv11 Trainer/Tester")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test_cam', action='store_true', help='Run webcam-based test')
    parser.add_argument('--test_image', type=str, help='Path to image for prediction')
    parser.add_argument('--data', type=str, help='Path to the dataset YAML file (for training)')
    parser.add_argument('--save_dir', type=str, help='Path to the result save directory')
    args = parser.parse_args()

    if args.train:
        print("[INFO] Starting training...")
        train_val_model(args.data,args.save_dir)
        print("[INFO] Training completed.")

    elif args.test_cam:
        print("[INFO] Starting real-time webcam test...")
        test_model_webcam()

    elif args.test_image:
        print(f"[INFO] Predicting image: {args.test_image}")
        predict_image(args.test_image)

    else:
        print("[WARNING] No action specified. Use --train, --test or --predict.")


if __name__ == "__main__":
    main()
