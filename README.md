# Face Recognition Security System

PyTorch와 OpenCV를 사용한 실시간 얼굴 인식 보안 시스템입니다. ResNet18 모델을 기반으로 특정 사용자를 인식하여 접근 권한을 부여하는 AI 시스템입니다.

## 주요 기능

- **실시간 얼굴 인식**: 웹캠 또는 비디오 파일을 통한 실시간 얼굴 감지 및 인식
- **스마트 보안 시스템**: 3단계 접근 제어 (OPEN/DENIED/UNKNOWN)
- **신뢰도 기반 판단**: 설정 가능한 임계값으로 정확한 인식
- **다중 얼굴 처리**: 화면 내 모든 얼굴을 동시에 감지하고 분류
- **실시간 성능 모니터링**: FPS 표시 및 확률 기반 결과 출력

## 프로젝트 구조

```
face_recog/
├── main.py                # 실시간 얼굴 인식 실행 파일
├── train.py               # 모델 학습 스크립트
├── collect_data.py        # 동영상 기반 데이터 수집 도구
├── preprocess_kaggle.py   # Kaggle 데이터 전처리
├── requirements.txt       # 필요한 패키지 목록
├── model/                 # 학습된 모델 저장 폴더
│   └── 20251209_052410/   # 실험 결과 (타임스탬프)
│       ├── face_model.pth # 학습된 모델 파일
│       ├── training_log.csv # 학습 로그
│       └── experiment_summary.txt # 실험 요약
└── dataset/              # 학습 데이터셋 폴더 ( 생성 필요 )
    ├── jisung/           # 개인 얼굴 이미지
    └── unknown/          # 일반인 얼굴 이미지
```
## Docker Environment

train 작업은 Docker 컨테이너 환경에서 실행하는 것을 권장합니다.
main.py를 통한 실시간 인식은 로컬 환경에서 실행하는 것을 권장합니다.

### Environment Spec
* **OS:** Ubuntu 22.04.4 LTS ( Jammy Jellyfish )
* **Python:** 3.11.9
* **CUDA:** 12.1 / **cuDNN:** 8.9.2
* **Framework:** PyTorch 2.2.2
* **Key Libraries:** OpenCV, NumPy (See `requirements.txt`)


## 설치 및 설정

### 1. 필요한 패키지 설치

```bash
pip install -r requirements.txt
```

### 2. 데이터셋 준비

#### 개인 데이터 수집
```python
# collect_data.py 설정 수정
LABEL = 'your_name'  # 본인 이름으로 변경
VIDEO_SOURCE = 0     # 웹캠 사용 또는 'video.mp4'
SAVE_COUNT = 300     # 수집할 이미지 개수
```

```bash
python collect_data.py
```

#### Unknown 데이터 준비
Kaggle의 얼굴 데이터셋을 다운로드하고 전처리:
```bash
python preprocess_kaggle.py
```

### 3. 모델 학습

```python
# train.py 설정 확인
DATA_DIR = './dataset'     # 데이터셋 경로
NUM_EPOCHS = 15           # 학습 에포크
BATCH_SIZE = 32           # 배치 크기
LEARNING_RATE = 0.0001    # 학습률
```

```bash
python train.py
```

### 4. 실시간 인식 실행

main.py 상단의 설정 섹션을 수정하세요:

```python
# ===============================================================
# [1] 사용자 설정
# ===============================================================
VIDEO_SOURCE = 0                    # 웹캠(0) 또는 비디오 파일('video.mp4')
CLASS_NAMES = ["jisung", "unknown"] # 학습한 클래스명 (알파벳 순서 중요!)
AUTHORIZED_USERS = ["jisung"]       # 접근 허용 사용자 목록
CONFIDENCE_THRESHOLD = 0.8          # 신뢰도 임계값 (0.7~0.8 권장)
MODEL_PATH = "./model/20251209_052410/face_model.pth"  # 모델 파일 경로
```

```bash
python main.py
```

**종료**: 'q' 키를 눌러 프로그램을 종료할 수 있습니다.

## 사용 방법

### 1. 데이터 수집
- `collect_data.py`를 실행하여 본인의 얼굴 데이터를 수집합니다
- 다양한 각도와 조명에서의 영상을 통해, 300장 정도 수집하는 것을 권장합니다
- `preprocess_kaggle.py`를 통해 Unknown 데이터를 확보합니다.

### 2. 모델 학습
- `train.py`를 실행하여 얼굴 인식 모델을 학습합니다
- 학습 결과는 `model/` 폴더에 타임스탬프와 함께 저장됩니다

### 3. 실시간 보안 시스템 실행
- `main.py`를 실행하여 AI 얼굴 보안 시스템을 시작합니다
- 시스템은 다음과 같이 동작합니다:

#### 인식 결과 표시
- **OPEN**: 등록된 사용자 인식 시 (초록색 박스)
- **DENIED**: 등록되지 않은 사용자 인식 시 (빨간색 박스)  
- **UNKNOWN**: 신뢰도가 임계값 미만이거나 모르는 사람 (빨간색 박스)

#### 실시간 정보 표시
- 각 얼굴마다 인식 결과와 신뢰도 퍼센트 표시
- 화면 좌상단에 실시간 FPS 표시
- 다중 얼굴 동시 처리 가능

## 모델 성능

현재 학습된 모델 정보:
- **모델**: ResNet18 (ImageNet 사전 훈련)
- **학습 에포크**: 15
- **배치 크기**: 32
- **학습률**: 0.0001
- **옵티마이저**: Adam
- **클래스**: jisung, unknown

## 기술 스택

- **딥러닝**: PyTorch, torchvision
- **컴퓨터 비전**: OpenCV, PIL
- **얼굴 감지**: MTCNN (facenet-pytorch)
- **모델**: ResNet18 (Transfer Learning)
- **데이터 처리**: NumPy, scikit-learn
- **시각화**: Matplotlib

## 시스템 아키텍처

### 실시간 추론 파이프라인 (main.py)

1. **얼굴 감지**: MTCNN으로 프레임에서 모든 얼굴 위치 탐지
2. **전처리**: 감지된 얼굴을 224x224로 리사이즈 및 정규화
3. **AI 추론**: ResNet18 모델로 얼굴 특성 추출 및 분류
4. **확률 변환**: Softmax로 신뢰도 점수를 확률로 변환
5. **접근 제어**: 임계값 기반 3단계 보안 판단
6. **시각화**: 실시간 결과 표시 및 FPS 모니터링

### 핵심 알고리즘

- **Transfer Learning**: ImageNet 사전훈련 → 얼굴 인식 파인튜닝
- **Confidence Thresholding**: 설정 가능한 임계값으로 Unknown 판별
- **Multi-face Processing**: 화면 내 모든 얼굴 동시 처리
- **Real-time Optimization**: torch.no_grad()로 추론 속도 최적화

## 주의사항

1. **GPU 사용**: CUDA가 설치된 환경에서 더 빠른 처리 가능
2. **조명 조건**: 다양한 조명에서 데이터를 수집하면 성능 향상
3. **데이터 품질**: 흐릿하거나 각도가 심한 이미지는 성능에 악영향
4. **개인정보**: 수집된 얼굴 데이터는 개인정보이므로 적절히 관리 필요
5. **임계값 조정**: CONFIDENCE_THRESHOLD를 0.7~0.8 범위에서 조정하여 최적 성능 확보
6. **클래스 순서**: CLASS_NAMES는 학습 시 데이터셋 폴더의 알파벳 순서와 정확히 일치해야 함

## 문제 해결

### 일반적인 오류

1. **모델 로딩 실패**
   - `MODEL_PATH` 경로가 올바른지 확인
   - `CLASS_NAMES`가 학습 시 클래스와 정확히 일치하는지 확인
   - 클래스 개수가 모델과 맞는지 확인

2. **카메라 열기 실패**
   - 웹캠이 다른 프로그램에서 사용 중인지 확인
   - `VIDEO_SOURCE` 값 확인 (0, 1, 2... 또는 파일 경로)
   - 권한 문제로 카메라 접근이 차단되었는지 확인

3. **CUDA 오류**
   - PyTorch CUDA 버전과 시스템 CUDA 버전 호환성 확인
   - GPU 메모리 부족 시 배치 크기 조정

4. **인식 성능 문제**
   - `CONFIDENCE_THRESHOLD` 값 조정 (0.7~0.8 권장)
   - 조명 조건 개선
   - 얼굴이 너무 작거나 각도가 심하지 않은지 확인

5. **Unknown으로만 인식되는 경우**
   - 임계값이 너무 높게 설정되었는지 확인
   - 학습 데이터와 실제 환경의 차이가 큰지 확인
   - 모델 재학습 고려

## 라이선스

이 프로젝트는 개인 학습 및 연구 목적으로 제작되었습니다.

## 기여

버그 리포트나 기능 개선 제안은 언제든 환영합니다.