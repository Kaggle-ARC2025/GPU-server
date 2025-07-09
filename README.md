# ARC Prize 2025 Solution - 로컬 실행 버전

이 저장소는 ARC Prize 2025 Kaggle 대회의 1등 솔루션을 로컬 환경에서 실행할 수 있도록 수정한 버전입니다.

## 개요

- **원본**: ARC Prize 2025 Kaggle Competition 1등 솔루션
- **모델**: 사전 훈련된 언어 모델 (wb55l_nemomini_fulleval)을 ARC 태스크에 파인튜닝
- **방법**: LoRA를 사용한 효율적인 파인튜닝 + 데이터 증강 + 터보 DFS 추론

## 필요 사항

- Python 3.8 이상
- CUDA 지원 GPU (최소 1개, 최대 4개)
- 16GB 이상의 GPU 메모리 (GPU당)
- 약 50GB의 디스크 공간

## 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd arc-prize-2025-solution
```

### 2. 환경 설정
```bash
# 실행 권한 부여
chmod +x setup.sh

# 설정 스크립트 실행
./setup.sh

# 가상환경 생성 및 활성화 (권장)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 3. 데이터 파일 준비

다음 파일들을 다운로드하여 지정된 위치에 복사:

1. **ARC 데이터셋** (Kaggle에서 다운로드)
   - `arc-agi_test_challenges.json` → `input/arc-prize-2025/`
   - `arc-agi_training_solutions.json` → `input/arc-prize-2025/`

2. **사전 훈련 모델** (HuggingFace 또는 Kaggle에서 다운로드)
   - 모델 파일들 → `input/wb55l_nemomini_fulleval/transformers/default/1/`
   - 필요 파일: `config.json`, `pytorch_model.bin`, `tokenizer.json` 등

3. **Unsloth** (선택사항)
   - wheel 파일들 → `input/unsloth-2024-9-post4/wheelhouse/`
   - 없으면 pip로 자동 설치됩니다

## 실행 방법

```bash
python main_local.py
```

## 설정 수정

`common_stuff.py`에서 다음 설정들을 조정할 수 있습니다:

### GPU 설정
```python
multi_gpu_train = True  # 다중 GPU 사용 여부
multi_gpu_random_split = True  # GPU별 데이터 무작위 분할
```

### 훈련 설정
```python
train_epochs = 4  # 훈련 에포크 수
max_seq_length_train = 4224  # 최대 시퀀스 길이
```

### 추론 설정
```python
max_seq_length_infer = 8192  # 추론 시 최대 시퀀스 길이
infer_params = dict(
    min_prob=0.17,  # 터보 DFS 최소 확률
    use_turbo=True  # 터보 모드 사용
)
```

### 메모리 부족 시
- `max_seq_length_train` 감소
- `per_device_train_batch_size` 감소 (start_training 함수 내)
- `train_epochs` 감소

## 디렉토리 구조

```
arc-prize-2025-solution/
├── input/
│   ├── arc-prize-2025/          # ARC 데이터셋
│   ├── wb55l_nemomini_fulleval/ # 사전 훈련 모델
│   └── unsloth-2024-9-post4/    # Unsloth wheel 파일
├── temp/                        # 임시 파일 및 체크포인트
├── model_runner.py              # 모델 실행 관련 함수
├── arc_loader.py                # ARC 데이터 로더
├── selection.py                 # 결과 선택 알고리즘
├── async_tools.py               # 비동기 도구
├── common_stuff.py              # 공통 설정 및 함수
├── main_local.py                # 로컬 실행 메인 스크립트
└── submission.json              # 최종 제출 파일
```

## 테스트 모드

빠른 테스트를 위해 가짜 테스트 세트를 사용할 수 있습니다:

```python
# common_stuff.py에서
arc_test_set.is_fake = True  # 가짜 테스트 세트 사용
```

## 주의사항

1. **GPU 메모리**: 최소 16GB GPU 메모리 필요
2. **디스크 공간**: 모델과 중간 결과 저장을 위해 충분한 공간 필요
3. **실행 시간**: 전체 실행에 수 시간이 소요될 수 있음
4. **Python 버전**: 3.8 이상 권장

## 문제 해결

### CUDA Out of Memory
- 배치 크기 감소
- 시퀀스 길이 감소
- 데이터 증강 옵션 감소

### 모델 로드 실패
- 모델 파일 경로 확인
- 필요한 모든 파일이 있는지 확인
- 파일 권한 확인

### Unsloth 설치 실패
- CUDA 버전 호환성 확인
- gcc 컴파일러 설치 확인
- pip로 직접 설치 시도

## 라이선스

원본 코드의 라이선스를 따릅니다 (Apache License 2.0).
