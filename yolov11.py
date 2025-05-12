import argparse
import cv2
from ultralytics import YOLO
import pandas as pd
import numpy as np
import os

def train_val_model(data_path, save_dir):
    # YOLO 모델 초기화 (학습 전: 초기 모델 경로, 학습 후: 현재 학습된 모델 경로)
    model = YOLO('/workspace/seonghyeon/mobility/runs/detect/train/weights/best.pt')

   
    # 모델 학습 수행
    results_train = model.train(
        data=data_path,         # 데이터셋 YAML 파일 경로
        epochs=20,              # 전체 데이터셋에 대한 학습 반복 횟수
        imgsz=640,              # 입력 이미지 크기 (픽셀)
        device=0                # 학습에 사용할 디바이스 ('cpu': CPU, 0: 첫 번째 GPU, [0,1,2,3]: 여러 GPU)
    )
    
    # 모델 검증 수행
    results_val = model.val(
        data=data_path,         # 데이터셋 YAML 파일 경로
        split="test",           # 검증 데이터셋 분할 (e.g., "test")
        save_json=True,         # 검증 결과를 JSON 파일로 저장 (True 또는 False)
        project=save_dir,       # 검증 결과 저장 디렉토리
        name="validation",      # 저장 폴더 이름
        imgsz=640,              # 이미지 크기
        batch=1,                # 배치 크기 (사용자의 컴퓨팅 자원에 맞게 설정)
        device=0,           # 검증에 사용할 디바이스 ('cpu': CPU, 0: 첫 번째 GPU, [0,1,2,3]: 여러 GPU)
        plots=True              # 플롯 생성 여부
    )

    # 클래스별 메트릭 추출
    metrics = {
        'ap50': results_val.box.ap50,  # AP@0.5
        'ap': results_val.box.ap,      # AP@0.5:0.95
        'precision': results_val.box.p,  # 정밀도
        'recall': results_val.box.r,     # 재현율
        'f1': results_val.box.f1         # F1 점수
    }
    
    # 전체 모델 성능 메트릭
    overall_metrics = {
        'mAP@0.5': results_val.box.map50,      # mAP@0.5
        'mAP@0.5:0.95': results_val.box.map,   # mAP@0.5:0.95
        'mean_precision': results_val.box.mp,  # 평균 정밀도
        'mean_recall': results_val.box.mr      # 평균 재현율
    }

    # 클래스 이름 매핑
    class_names = model.names  

    # 클래스별 메트릭 DataFrame 생성
    class_metrics = []
    for i in range(len(class_names)):
        class_metric = {
            'Class Name': class_names[i],  # 클래스 이름
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
    for folder in os.listdir(save_dir):
        if folder.startswith('validation'):
            validation_folder = folder
    excel_dir = os.path.join(save_dir, validation_folder)
    # 결과를 엑셀 파일로 저장
    with pd.ExcelWriter(excel_dir + '/detection_metrics_results.xlsx') as writer:
        df_class_metrics.to_excel(writer, sheet_name='Class Metrics', index=False)
        df_overall_metrics.to_excel(writer, sheet_name='Overall Metrics', index=False)

    # 모델 평가 결과 출력
    print("\n=== Overall Detection Metrics ===")
    for k, v in overall_metrics.items():
        print(f"{k}: {v:.4f}")

    return results_val

def test_model_webcam():
    # YOLO 모델 초기화
    model = YOLO('D:\\mobility_experiments\\best.pt')

    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] 웹캠을 열 수 없습니다.")
        return

    print("[INFO] 'q' 키를 눌러 종료하세요.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 모델로 프레임 예측
        results = model.predict(source=frame, imgsz=640, conf=0.3, stream=True) # 프레임, 이미지 크기, 신뢰도

        # 예측 결과를 플롯하여 표시
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow('YOLOv11 웹캠 객체 검출', annotated_frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def predict_image(image_path):
    # YOLO 모델 초기화
    model = YOLO('users_dir/yolo11n.pt')
    # 이미지 예측
    results = model.predict(source=image_path, imgsz=640, conf=0.3) # 이미지 경로, 이미지 크기, 신뢰도

    # 예측 결과를 플롯하여 표시
    for r in results:
        annotated = r.plot()
        cv2.imshow("YOLOv11 이미지 객체 검출", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return results


def main():
    # 명령줄 인자 파서 설정
    parser = argparse.ArgumentParser(description="YOLOv11 학습/테스트")
    parser.add_argument('--train', action='store_true', help='모델 학습')
    parser.add_argument('--test_cam', action='store_true', help='웹캠 테스트')
    parser.add_argument('--test_image', type=str, help='예측할 이미지 경로')
    parser.add_argument('--data', type=str, help='데이터셋 YAML 파일 경로')
    parser.add_argument('--save_dir', type=str, help='결과 저장 디렉토리 경로')
    args = parser.parse_args()

    if args.train:
        print("[INFO] 학습을 시작합니다...")
        train_val_model(args.data, args.save_dir)
        print("[INFO] 학습이 완료되었습니다.")

    elif args.test_cam:
        print("[INFO] 실시간 웹캠 테스트를 시작합니다...")
        test_model_webcam()

    elif args.test_image:
        print(f"[INFO] 이미지 예측 중: {args.test_image}")
        predict_image(args.test_image)

    else:
        print("[WARNING] 실행할 작업이 지정되지 않았습니다. --train, --test_cam 또는 --test_image를 사용하세요.")


if __name__ == "__main__":
    main()
