# common_stuff.py
# 전체 내용은 원본 노트북의 %%writefile common_stuff.py 셀에서 복사해주세요.
# 이 파일은 공통 설정과 메인 함수들을 포함합니다.

# ARC 훈련 및 평가를 위한 공통 설정
import os
import time
from arc_loader import *
from model_runner import *
from selection import *
from async_tools import *

# ===== 파일 경로 설정 =====
tmp_dir = '/kaggle/temp'  # 임시 파일 저장 디렉토리
arc_challenge_file = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'  # ARC 테스트 문제 파일
arc_solutions_file = '/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json'  # ARC 훈련 솔루션 파일
model_temp_storage = os.path.join(tmp_dir, 'finetuned_model')  # 파인튜닝된 모델 저장 경로
infer_temp_storage = os.path.join(tmp_dir, 'inference_outputs')  # 추론 결과 저장 경로
score_temp_storage = os.path.join(tmp_dir, 'inference_scoring')  # 점수 계산 결과 저장 경로

# ===== 데이터셋 로드 (지연 로딩) =====
arc_test_set = None  # 필요할 때 로드됩니다

def load_arc_dataset():
    """ARC 데이터셋을 로드하는 함수"""
    global arc_test_set
    if arc_test_set is None:
        arc_test_set = ArcDataset.from_file(arc_challenge_file)  # ARC 테스트 세트 로드
        if arc_test_set.is_fake: arc_test_set.load_replies(arc_solutions_file)  # 가짜 테스트 세트인 경우 솔루션 로드
    return arc_test_set
#arc_test_set.is_fake = False  # 전체 실행 강제 (주석 처리됨)
#arc_train_set = ArcDataset.from_file('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json')  # 훈련 세트 (사용 안함)

# ===== 모델 설정 =====
base_model = '/kaggle/input/wb55l_nemomini_fulleval/transformers/default/1'  # 기본 모델 경로
MyFormatter = ArcFormatter_premix_3  # 사용할 포맷터 (수학 기호 포함 버전)
perm_aug = 'rnd_all'  # 순열 증강 타입 (모든 색상 무작위 순열)
max_seq_length_train = 4224  # 훈련 시 최대 시퀀스 길이
mask_first = 0  # 첫 번째 출력 마스킹 설정

# ===== 훈련 및 추론 설정 =====
train_epochs = 4  # 훈련 에포크 수
multi_gpu_train = True  # 다중 GPU 훈련 사용 여부
multi_gpu_random_split = True  # 다중 GPU 시 무작위 분할 여부
max_seq_length_infer = 8192  # 추론 시 최대 시퀀스 길이
prime_on_single_task = False  # 단일 작업 프라이밍 여부
infer_params = dict(
    min_prob=0.17,  # 터보 DFS 최소 확률 임계값
    store=infer_temp_storage,  # 추론 결과 저장 경로
    use_turbo=True  # 터보 모드 사용 여부
)

# ===== 점수 계산 설정 =====
use_aug_score = True  # 증강 점수 사용 여부
aug_score_params = dict(
    tp=True,  # 전치 변환 사용
    rot=True,  # 회전 변환 사용
    perm=perm_aug,  # 순열 증강 타입
    shfl_ex=True,  # 예제 셔플 사용
    make_unique=True,  # 고유성 확보
    max_len=max_seq_length_infer  # 최대 길이
)
# 제출용 선택 알고리즘 (증강 점수 사용 여부에 따라 결정)
submission_select_algo = score_full_probmul_3 if use_aug_score else score_all_probsum

def prepare_run(model_path, load_lora=None, train=False, gpu=None, **kwargs):
    """
    모델 실행을 위한 준비 함수
    
    Args:
        model_path: 모델 경로
        load_lora: 로드할 LoRA 경로
        train: 훈련 모드 여부
        gpu: 사용할 GPU 번호
        **kwargs: 추가 인자들
    
    Returns:
        (모델, 포맷터) 튜플
    """
    # GPU 설정
    if gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"   ] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # 모델, 토크나이저, 포맷터 준비
    model, tokenizer, formatter = prepare_model(
        model=model_path,
        local_files_only=True,  # 로컬 파일만 사용
        mode='unsloth_4bit',  # Unsloth 4비트 모드
        #shrink_embedding=8000,  # 임베딩 축소 (주석 처리됨)
        max_seq_length=max_seq_length_train,
        formatter=MyFormatter,
        # LoRA 설정 (훈련 또는 LoRA 로드 시에만)
        peft=([dict(
            r=64,  # LoRA 랭크 (8, 16, 32, 64, 128 중 선택)
            target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'embed_tokens', 'lm_head'],
            lora_alpha=16,  # LoRA 알파 값
            lora_dropout=0,  # LoRA 드롭아웃 (0이 최적화됨)
            bias="none",  # 바이어스 설정 ("none"이 최적화됨)
            use_gradient_checkpointing=True,  # 그래디언트 체크포인팅 사용
            random_state=42,  # 랜덤 시드
            use_rslora=True,  # 랭크 안정화 LoRA 사용
            loftq_config=None,  # LoftQ 설정
        )] if train or load_lora else []) + ([load_lora] if load_lora else []),
        **kwargs
    )
    
    # 훈련 시 첫 번째 출력 마스킹 설정
    if train and mask_first: 
        formatter.collator_kwargs.update(mask_first_output=mask_first)

    return model, formatter

def prepare_dataset(formatter, train, gpu=None):
    """
    데이터셋을 준비하는 함수
    
    Args:
        formatter: 포맷터 객체
        train: 훈련 모드 여부
        gpu: 사용할 GPU 번호
    
    Returns:
        준비된 ArcDataset
    """
    ds = load_arc_dataset()
    
    # 다중 GPU 훈련 시 데이터 분할
    if multi_gpu_train and gpu is not None:
        if multi_gpu_random_split:
            # 4개 GPU용 무작위 분할
            ds = ds.shuffled(seed=123)
            split_size = len(ds.keys) // 4
            start_idx = gpu * split_size
            end_idx = start_idx + split_size if gpu < 3 else len(ds.keys)
            ds = ds.change_keys(ds.keys[start_idx:end_idx])
        else:
            # 길이 기반 분할 (4개 GPU용)
            ds = ds.sorted_by_len(formatter=formatter, name='input', max_of_transposed=True)
            assignment = ([0,1,2,3]*ds.length())[:ds.length()][::-1]
            ds = ds.change_keys((np.array(ds.keys)[np.array(assignment)==gpu]).tolist())
    
    
    if train:
        # ===== 훈련용 데이터셋 준비 =====
        ds = ds.remove_replies()  # 답안 제거 (자가 지도 학습)
        # 데이터 증강
        ds = ds.augment(
            tp=True,  # 전치 변환
            rot=True,  # 회전 변환
            perm=perm_aug,  # 순열 증강
            n=(2 if load_arc_dataset().is_fake else train_epochs),  # 증강 횟수
            shfl_ex=True,  # 예제 셔플
            shfl_keys=True  # 키 셔플
        )
        # 최대 길이로 자르기 (새 토큰 생성 없음)
        ds = ds.cut_to_len(formatter=formatter, name='text', max_len=max_seq_length_train, max_new_tokens=0)
        # 가짜 테스트 세트인 경우 긴 것부터 정렬
        if load_arc_dataset().is_fake: 
            ds = ds.sorted_by_len(formatter=formatter, name='text', reverse=True)
    else:
        # 추론 시에도 4개 GPU로 분할
        ds = ds.sorted_by_len(formatter=formatter, name='input', max_of_transposed=True)
        ds = ds.split_multi_replies()
        
        # 4개 GPU로 균등 분할
        if gpu is not None:
            total_keys = len(ds.keys)
            split_size = total_keys // 4
            start_idx = gpu * split_size
            end_idx = start_idx + split_size if gpu < 3 else total_keys
            ds.keys = ds.keys[start_idx:end_idx]
        
        # 증강 및 인터리브
        ds = ds.augment(tp=True, rot=True, n=2, seed=42, perm=perm_aug, shfl_ex=True).interleave(len(ds.keys))
        ds = ds.cut_to_len(formatter=formatter, name='input', max_len=max_seq_length_infer)
        
        if load_arc_dataset().is_fake: 
            ds.keys = ds.keys[:32]  # 각 GPU당 32개씩
    
    return ds

def start_training(gpu):
    """
    지정된 GPU에서 훈련을 시작하는 함수
    
    Args:
        gpu: 사용할 GPU 번호
    """
    try:
        storage_path = f'{model_temp_storage}_gpu{gpu}'
        # GPU 0이거나 다중 GPU 모드이고, 저장 경로가 없는 경우에만 훈련
        if (gpu==0 or multi_gpu_train) and not os.path.exists(storage_path):
            with RemapCudaOOM():  # CUDA OOM 에러 처리
                # 모델과 포맷터 준비
                model, formatter = prepare_run(base_model, train=True, gpu=gpu)
                # 데이터셋 준비
                dataset = prepare_dataset(formatter, train=True, gpu=gpu if multi_gpu_train else None)
                
                # 훈련 실행
                model, trainer_stats = training_run(
                    model, formatter, dataset, 
                    store=storage_path,  # 모델 저장 경로
                    max_seq_length=max_seq_length_train,
                    grad_acc_fix=False,  # 그래디언트 누적 수정 비활성화
                    train_args=dict(
                        per_device_train_batch_size=2,  # 디바이스당 배치 크기
                        gradient_accumulation_steps=2,  # 그래디언트 누적 단계
                        warmup_steps=100,  # 워밍업 단계
                        num_train_epochs=1,  # 훈련 에포크 수
                        max_steps=20 if load_arc_dataset().is_fake else -1,  # 최대 스텝 (가짜 세트는 20스텝만)
                        learning_rate=1e-4,  # 학습률
                        embedding_learning_rate=1e-5,  # 임베딩 학습률
                        logging_steps=10,  # 로깅 간격
                        optim="adamw_8bit",  # 8비트 AdamW 옵티마이저
                        weight_decay=0.01,  # 가중치 감쇠
                        lr_scheduler_type='cosine',  # 코사인 학습률 스케줄러
                        seed=42,  # 랜덤 시드
                        output_dir=os.path.join(tmp_dir, 'checkpoints'),  # 체크포인트 저장 경로
                        save_strategy="no",  # 저장 전략 (저장 안함)
                        report_to='none',  # 리포팅 비활성화
                    ),
                )
                mem_info()  # 메모리 정보 출력
    finally: 
        # 훈련 완료 표시 파일 생성
        os.makedirs(f'{storage_path}_done', exist_ok=True)

def start_inference(gpu):
    """
    지정된 GPU에서 추론을 시작하는 함수
    
    Args:
        gpu: 사용할 GPU 번호
    """
    storage_path = f'{model_temp_storage}_gpu{gpu % 4 if multi_gpu_train else 0}'

    
    # 훈련 완료까지 대기
    while not os.path.exists(f'{storage_path}_done'): 
        time.sleep(15)
    
    with RemapCudaOOM():  # CUDA OOM 에러 처리
        # 훈련된 모델로 준비
        model, formatter = prepare_run(storage_path, gpu=gpu)
        # 추론용 데이터셋 준비
        dataset = prepare_dataset(formatter, train=False, gpu=gpu)
        
        # 단일 작업 프라이밍을 위한 재훈련기 설정
        retrainer = None if not prime_on_single_task else Retrainer(
            n=32,  # 훈련 샘플 수
            aug_opts=dict(perm=perm_aug, shfl_ex=True),  # 증강 옵션
            reload_state_dict=get_and_fix_peft_weights(storage_path),  # PEFT 가중치 재로드
            formatter=formatter,
            max_seq_length=max_seq_length_infer,
            grad_acc_fix=False,
            train_args=dict(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                warmup_steps=4,
                num_train_epochs=1,
                learning_rate=1e-4,
                embedding_learning_rate=0,  # 임베딩 학습률 0 (고정)
                logging_steps=8,
                optim="adamw_8bit",
                weight_decay=0.00,  # 가중치 감쇠 없음
                lr_scheduler_type='constant',  # 상수 학습률
                seed=42,
                output_dir='tmp_output',
                save_strategy='no',
                report_to='none',
            ),
        )
        
        # 디코더 설정
        decoder = Decoder(
            formatter, 
            load_arc_dataset().split_multi_replies(), 
            n_guesses=2,  # 최대 추측 횟수
            prob_baseline=0.05  # 확률 기준선
        )
        
        # 추론 실행
        inference_run_v2(model, formatter, dataset, decoder, retrain=retrainer, **infer_params)
        
        # 증강 점수 계산 (필요한 경우)
        if use_aug_score or load_arc_dataset().is_fake: 
            decoder.calc_augmented_scores(model=model, store=score_temp_storage, **aug_score_params)
        
        mem_info()  # 메모리 정보 출력

class RemapCudaOOM:
    """CUDA Out of Memory 에러를 처리하는 컨텍스트 매니저"""
    
    def __enter__(self): 
        """컨텍스트 진입 시 아무것도 하지 않음"""
        pass
    
    def __exit__(self, exc_type, exc_value, traceback):
        """
        컨텍스트 종료 시 CUDA OOM 에러 처리
        
        CUDA 메모리 부족 에러가 발생하면 제출 파일을 생성하여
        채점 에러를 발생시킴 (Kaggle 환경에서의 안전장치)
        """
        oom_errors = [
            "CUDA out of memory", 
            "Make sure you have enough GPU RAM", 
            "does not fit any GPU's remaining memory"
        ]
        if exc_value and any(x in str(exc_value) for x in oom_errors):
            # 의도적으로 잘못된 제출 파일 생성
            with open('submission.json', 'w') as f: 
                f.write('cause submission scoring error')