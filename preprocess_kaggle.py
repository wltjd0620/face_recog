import os
import random
from PIL import Image
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN
import numpy as np

# ==================================================================
# [설정]
SOURCE_DIR = r'/workspace/face_recog/dataset_make_video/Humans'  # Kaggle 데이터 경로
DEST_DIR = r'/workspace/face_recog/dataset/unknown'         # 저장할 경로
TARGET_COUNT = 450                      # 목표 개수
# ==================================================================

def preprocess_images_clean():
    # 1. 장치 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"색상 왜곡 없는 전처리 시작 (Device: {device})")

    # 2. MTCNN 설정
    # keep_all=False: 가장 확률 높은 얼굴 1개만 찾음
    # post_process=False: (중요) 수학 계산 안 함!
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)

    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
    
    # 이미지 목록 수집
    print("이미지 파일 수집 중...")
    all_images = []
    for root, dirs, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    
    random.shuffle(all_images)
    print(f"총 {len(all_images)}장 중 {TARGET_COUNT}장을 처리합니다.")

    count = 0
    try: iterator = tqdm(all_images)
    except ImportError: iterator = all_images

    for img_path in iterator:
        if count >= TARGET_COUNT: break

        try:
            # 1. 이미지 열기 및 RGB 변환 (투명도 에러 방지)
            img = Image.open(img_path).convert('RGB')
            
            # 2. [핵심 변경] 얼굴 위치(Box)만 받아오기
            boxes, _ = mtcnn.detect(img)

            if boxes is not None:
                # 가장 큰 얼굴 하나만 선택
                box = boxes[0] 
                
                # 좌표를 정수로 변환 (PIL Crop을 위해)
                x1, y1, x2, y2 = [int(b) for b in box]
                
                # 좌표가 이미지 밖으로 나가지 않게 조절
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img.width, x2)
                y2 = min(img.height, y2)

                # 3. [핵심] 원본 이미지에서 직접 자르기 (색상 왜곡 X)
                face_img = img.crop((x1, y1, x2, y2))
                
                # 4. 224x224로 리사이징
                face_img = face_img.resize((224, 224))

                # 5. 저장
                save_name = f"unknown_{count+1:04d}.jpg"
                save_path = os.path.join(DEST_DIR, save_name)
                face_img.save(save_path, 'JPEG', quality=95)
                
                count += 1
                
        except Exception as e:
            pass

    print("------------------------------------------------")
    print(f"✅ 완료! 총 {count}장의 '정상 색깔' 이미지가 저장되었습니다.")

if __name__ == '__main__':
    preprocess_images_clean()