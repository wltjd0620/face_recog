import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import datetime
import csv
import pandas as pd
from sklearn.metrics import precision_score, recall_score
# [추가] 그래프 그리기용 라이브러리
import matplotlib.pyplot as plt

# ====================================================================
# [1] 설정 (Hyperparameters)
# ====================================================================
DATA_DIR = '/workspace/face_recog/dataset' # 데이터 폴더 경로

# 현재 시간으로 폴더 이름 생성
now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
base_save_dir = '/workspace/face_recog/model'
SAVE_DIR = os.path.join(base_save_dir, now_str) # 예: model/20251208_173000

# 파일 저장 경로 설정
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'face_model.pth')
LOG_CSV_PATH = os.path.join(SAVE_DIR, 'training_log.csv')
SUMMARY_PATH = os.path.join(SAVE_DIR, 'experiment_summary.txt')
# [추가] 그래프 이미지 저장 경로
GRAPH_PATH = os.path.join(SAVE_DIR, 'training_metrics.png')

BATCH_SIZE = 32
LEARNING_RATE = 0.0001 # (튜닝된 값 추천)
NUM_EPOCHS = 15
TRAIN_SPLIT_RATIO = 0.8
SEED = 42 # 랜덤 시드 고정
# ====================================================================

# 랜덤 시드 고정 함수
def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# [추가] 그래프 그리기 및 저장 함수
def plot_metrics(df, save_path):
    plt.figure(figsize=(15, 5))

    # 1. Loss 그래프
    plt.subplot(1, 3, 1)
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss', marker='o')
    plt.plot(df['epoch'], df['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. Accuracy 그래프
    plt.subplot(1, 3, 2)
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc', marker='o', color='green')
    plt.plot(df['epoch'], df['val_acc'], label='Val Acc', marker='o', color='red')
    plt.title('Accuracy per Epoch (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 3. Precision 그래프
    plt.subplot(1, 3, 3)
    plt.plot(df['epoch'], df['train_precision'], label='Train Precision', marker='o', color='purple')
    plt.plot(df['epoch'], df['val_precision'], label='Val Precision', marker='o', color='orange')
    plt.title('Precision per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path) # 그래프를 이미지 파일로 저장
    plt.close() # 메모리 해제
    print(f"📊 학습 그래프 저장됨: {save_path}")

def train_model():
    # 0. 폴더 생성
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print(f"📁 실험 폴더 생성 완료: {SAVE_DIR}")

    set_seed(SEED)

    print("-------------------------")
    print(f"🚀 학습 시작 (Log: {now_str})")
    print("-------------------------")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"1. 학습 장치: {device}")

    # 1. 데이터 전처리
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2,contrast=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 2. 데이터셋 로드
    try:
        full_dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    except Exception as e:
        print(f"🚨 에러: 데이터를 못 찾겠습니다. ({e})")
        return
    
    train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"📂 데이터: Train {len(train_dataset)}장 / Val {len(val_dataset)}장")
    class_names = full_dataset.classes
    print(f"🏷️ 클래스: {class_names}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 모델 설계
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # (선택사항: 미세 조정 성능 향상을 위해 layer4 잠금 해제)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 4. 설정 저장 (Optimizer, Loss 등)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 실험 요약 파일 저장 (Summary.txt)
    with open(SUMMARY_PATH, 'w', encoding='utf-8') as f:
        f.write(f"Experiment Time: {now_str}\n")
        f.write(f"Model: ResNet18 (Layer4 Unfrozen)\n")
        f.write(f"Epochs: {NUM_EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write(f"Optimizer: Adam\n")
        f.write(f"Dataset Split: {TRAIN_SPLIT_RATIO} : {1-TRAIN_SPLIT_RATIO:.1f}\n")
        f.write(f"Classes: {class_names}\n")
        f.write("-" * 30 + "\n")
        f.write("Model Structure:\n")
        f.write(str(model))

    # 6. 로그 기록용 리스트
    log_history = []

    print(f"\n🔥 학습 루프 시작 ({NUM_EPOCHS} Epochs)")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        epoch_metrics = {'epoch': epoch + 1}

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader
            
            running_loss = 0.0
            correct = 0
            total = 0
            
            all_preds = []
            all_labels = []

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                correct += torch.sum(preds == labels.data)
                total += inputs.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / total
            epoch_acc = correct.double() / total * 100
            
            epoch_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)

            print(f'{phase.upper()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% '
                  f'Prec: {epoch_precision:.4f} Recall: {epoch_recall:.4f}')

            epoch_metrics[f'{phase}_loss'] = epoch_loss
            epoch_metrics[f'{phase}_acc'] = epoch_acc.item()
            epoch_metrics[f'{phase}_precision'] = epoch_precision
            epoch_metrics[f'{phase}_recall'] = epoch_recall

        log_history.append(epoch_metrics)

    time_elapsed = time.time() - start_time
    print(f'\n✅ 학습 완료! 소요 시간: {time_elapsed // 60:.0f}분 {time_elapsed % 60:.0f}초')

    # 7. 모델 저장
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 모델 저장됨: {MODEL_SAVE_PATH}")

    # 8. 로그(CSV) 저장
    df = pd.DataFrame(log_history)
    df.to_csv(LOG_CSV_PATH, index=False)
    print(f"📝 학습 로그 저장됨: {LOG_CSV_PATH}")

    # [추가] 9. 그래프 그리기 및 저장
    plot_metrics(df, GRAPH_PATH)
    
    print(f"👉 main.py의 CLASS_NAMES를 이걸로 바꾸세요: {class_names}")

if __name__ == "__main__":
    train_model()