#!/usr/bin/env python3
"""
ARC Prize 2025 로컬 실행을 위한 메인 스크립트
Kaggle 노트북을 로컬 환경에서 실행할 수 있도록 수정된 버전
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# 환경 변수 설정
os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# 로컬 경로 설정
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR))

# common_stuff.py의 경로들을 로컬용으로 오버라이드
import common_stuff

# 경로 재설정 (로컬 환경용)
common_stuff.tmp_dir = str(BASE_DIR / 'temp')
common_stuff.arc_challenge_file = str(BASE_DIR / 'input/arc-prize-2025/arc-agi_test_challenges.json')
common_stuff.arc_solutions_file = str(BASE_DIR / 'input/arc-prize-2025/arc-agi_training_solutions.json')
common_stuff.base_model = str(BASE_DIR / 'input/wb55l_nemomini_fulleval/transformers/default/1')
common_stuff.model_temp_storage = os.path.join(common_stuff.tmp_dir, 'finetuned_model')
common_stuff.infer_temp_storage = os.path.join(common_stuff.tmp_dir, 'inference_outputs')
common_stuff.score_temp_storage = os.path.join(common_stuff.tmp_dir, 'inference_scoring')

# 필요한 디렉토리 생성
os.makedirs(common_stuff.tmp_dir, exist_ok=True)
os.makedirs(os.path.join(common_stuff.tmp_dir, 'checkpoints'), exist_ok=True)

def check_files_exist():
    """필수 파일들이 존재하는지 확인"""
    required_files = [
        common_stuff.arc_challenge_file,
        # common_stuff.arc_solutions_file,  # 테스트 세트인 경우 선택사항
        common_stuff.base_model,
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("다음 파일들이 없습니다:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n필요한 파일들을 다운로드하고 올바른 경로에 배치해주세요.")
        return False
    return True

def install_unsloth_local():
    """로컬에서 unsloth 설치 (이미 설치되어 있지 않은 경우)"""
    try:
        import unsloth
        print("Unsloth가 이미 설치되어 있습니다.")
    except ImportError:
        print("Unsloth를 설치합니다...")
        wheelhouse = BASE_DIR / 'input/unsloth-2024-9-post4/wheelhouse'
        if wheelhouse.exists():
            # 로컬 wheel 파일에서 설치
            os.system(f"pip install --no-index --find-links={wheelhouse} unsloth")
        else:
            # pip에서 직접 설치
            os.system("pip install unsloth")

def run_single_gpu():
    """단일 GPU에서 실행"""
    print("단일 GPU 모드로 실행합니다...")
    
    # 훈련
    print("\n=== 훈련 시작 ===")
    common_stuff.start_training(gpu=0)
    
    # 추론
    print("\n=== 추론 시작 ===")
    common_stuff.start_inference(gpu=0)
    
    # 제출 파일 생성
    print("\n=== 제출 파일 생성 ===")
    create_submission()

def run_multi_gpu():
    """다중 GPU에서 실행 (비동기)"""
    print("다중 GPU 모드로 실행합니다...")
    
    async def run_all():
        # 훈련 태스크들
        train_tasks = []
        for gpu in range(4):
            task = asyncio.create_task(
                asyncio.to_thread(common_stuff.start_training, gpu=gpu)
            )
            train_tasks.append(task)
        
        # 모든 훈련 완료 대기
        print("\n=== 훈련 시작 (4개 GPU) ===")
        await asyncio.gather(*train_tasks)
        
        # 추론 태스크들
        infer_tasks = []
        for gpu in range(4):
            task = asyncio.create_task(
                asyncio.to_thread(common_stuff.start_inference, gpu=gpu)
            )
            infer_tasks.append(task)
        
        # 모든 추론 완료 대기
        print("\n=== 추론 시작 (4개 GPU) ===")
        await asyncio.gather(*infer_tasks)
    
    # 비동기 실행
    asyncio.run(run_all())
    
    # 제출 파일 생성
    print("\n=== 제출 파일 생성 ===")
    create_submission()

def create_submission():
    """제출 파일 생성"""
    from common_stuff import (
        load_arc_dataset, MyFormatter, Decoder, 
        infer_temp_storage, score_temp_storage,
        use_aug_score, aug_score_params, submission_select_algo,
        multi_gpu_train, RemapCudaOOM
    )
    
    arc_test_set = load_arc_dataset()
    
    with RemapCudaOOM():
        model, formatter, dataset = None, MyFormatter(), None
        decoder = Decoder(formatter, arc_test_set.split_multi_replies(), n_guesses=2, frac_score=True)
        
        # 모든 GPU의 결과를 병합
        if multi_gpu_train:
            for gpu in range(4):
                gpu_store = f"{infer_temp_storage}_gpu{gpu}"
                if os.path.exists(gpu_store):
                    decoder.from_store(gpu_store)
        else:
            if os.path.exists(infer_temp_storage):
                decoder.from_store(infer_temp_storage)
        
        if use_aug_score or arc_test_set.is_fake: 
            decoder.calc_augmented_scores(model=model, store=score_temp_storage, **aug_score_params)
        
        submission = arc_test_set.get_submission(decoder.run_selection_algo(submission_select_algo))
        
        # 제출 파일 저장
        submission_path = BASE_DIR / 'submission.json'
        with open(submission_path, 'w') as f:
            json.dump(submission, f)
        print(f"제출 파일이 생성되었습니다: {submission_path}")
        
        # 가짜 테스트 세트인 경우 점수 확인
        if arc_test_set.is_fake:
            from common_stuff import selection_algorithms
            decoder.benchmark_selection_algos(selection_algorithms)
            with open(submission_path) as f:
                reload_submission = json.load(f)
            score = arc_test_set.validate_submission(reload_submission)
            print(f'*** 검증 점수: {score:.2%}')

def main():
    """메인 실행 함수"""
    print("ARC Prize 2025 로컬 실행 스크립트")
    print("=" * 50)
    
    # 파일 확인
    if not check_files_exist():
        return
    
    # unsloth 설치 확인
    install_unsloth_local()
    
    # GPU 개수 확인
    import torch
    gpu_count = torch.cuda.device_count()
    print(f"\n사용 가능한 GPU 개수: {gpu_count}")
    
    if gpu_count == 0:
        print("GPU가 없습니다. CPU 모드는 지원하지 않습니다.")
        return
    
    # 설정 확인
    from common_stuff import multi_gpu_train, load_arc_dataset
    
    arc_test_set = load_arc_dataset()
    if arc_test_set.is_fake:
        print("가짜 테스트 세트 모드로 실행합니다 (빠른 테스트)")
    
    # 실행
    if multi_gpu_train and gpu_count >= 4:
        run_multi_gpu()
    else:
        if multi_gpu_train:
            print(f"경고: multi_gpu_train이 True이지만 GPU가 {gpu_count}개만 있습니다.")
            print("단일 GPU 모드로 전환합니다.")
            common_stuff.multi_gpu_train = False
        run_single_gpu()
    
    print("\n완료!")

if __name__ == "__main__":
    main()
