# model_runner.py
# 전체 내용은 원본 노트북의 %%writefile model_runner.py 셀에서 복사해주세요.
# 이 파일은 모델 실행, 훈련, 추론 관련 모든 함수들을 포함합니다.
import json
import os, sys
import bz2
import pickle
import random
import numpy as np
from tqdm import tqdm

def indices_required_for_merges(keep_indices, vocab, merges):
    """
    BPE 병합에 필요한 모든 토큰 인덱스를 찾는 함수
    
    Args:
        keep_indices: 유지할 토큰 인덱스들의 딕셔너리
        vocab: 어휘 사전
        merges: BPE 병합 규칙들
    
    Returns:
        병합에 필요한 모든 인덱스가 포함된 딕셔너리
    """
    merges_lookup = {}
    # 각 병합 규칙에서 필요한 하위 토큰들을 매핑
    for m in merges:
        a, b = m.split(' ') if isinstance(m, str) else m
        key = vocab[f'{a}{b}']
        if key not in merges_lookup: merges_lookup[key] = set()
        merges_lookup[key].add(vocab[a])
        merges_lookup[key].add(vocab[b])
    
    # 재귀적으로 필요한 모든 토큰 인덱스 수집
    to_process = list(keep_indices)
    while len(to_process):
        for w in merges_lookup.get(to_process.pop(), []):
            if w not in keep_indices:
                keep_indices[w] = None
                to_process.append(w)
    return keep_indices

def remove_unused_merges(merges, vocab):
    """
    사용되지 않는 BPE 병합 규칙들을 제거하는 함수
    
    Args:
        merges: 병합 규칙 리스트
        vocab: 어휘 사전
    
    Returns:
        유효한 병합 규칙들만 포함된 리스트
    """
    return [f'{a} {b}' for a, b in [m.split(' ') if isinstance(m, str) else m for m in merges] 
            if all(w in vocab for w in [a, b, a + b])]

def map_special_tokens(data, mapping=None):
    """
    특별 토큰들을 매핑하거나 수집하는 함수
    
    Args:
        data: 토큰 데이터 (딕셔너리 또는 리스트)
        mapping: 토큰 인덱스 매핑 (선택사항)
    
    Returns:
        특별 토큰 인덱스들의 집합
    """
    tokens = set()
    if isinstance(data, dict):
        special = data.get('special_tokens')
        if special is not None:
            for v in special.values():
                tokens.update(v['ids'])
                # 매핑이 제공된 경우 토큰 ID들을 새로운 인덱스로 변환
                if mapping is not None:
                    v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
    
    # 재귀적으로 중첩된 데이터 구조 처리
    for v in (data.values() if isinstance(data, dict) else data if isinstance(data, list) else []):
        tokens.update(map_special_tokens(v, mapping))
    return tokens

def remove_tokenizer_normalizer(tokenizer):
    """
    토크나이저의 정규화 기능을 제거하는 함수
    
    Args:
        tokenizer: HuggingFace 토크나이저
    """
    from tokenizers import Tokenizer
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    if tokenizer_json.get('normalizer') is not None:
        tokenizer_json['normalizer'] = None
        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

def shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special_tokens, keep_token_order):
    """
    토크나이저의 어휘를 축소하는 함수
    
    Args:
        tokenizer: 토크나이저 객체
        keep_indices: 유지할 토큰 인덱스들
        keep_special_tokens: 특별 토큰 유지 여부
        keep_token_order: 토큰 순서 유지 여부
    
    Returns:
        (매핑 딕셔너리, 유지된 인덱스들)
    """
    from tokenizers import Tokenizer
    assert tokenizer.is_fast
    tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
    assert tokenizer_json['model']['type'] == "BPE"
    
    # 특별 토큰들 추가
    if keep_special_tokens:
        keep_indices.update({k: None for k in tokenizer.all_special_ids})
        keep_indices.update({k: None for k in map_special_tokens(tokenizer_json.get('post_processor'))})
    
    # BPE 병합에 필요한 모든 인덱스 포함
    keep_indices = indices_required_for_merges(keep_indices, tokenizer_json['model']['vocab'], tokenizer_json['model']['merges'])
    
    # 토큰 순서 정렬
    if keep_token_order: keep_indices = sorted(keep_indices)
    
    # 새로운 인덱스 매핑 생성
    mapping = {old: new for new, old in enumerate(keep_indices)}
    
    # 어휘 사전 업데이트
    tokenizer_json['model']['vocab'] = {k: mapping[v] for k, v in tokenizer_json['model']['vocab'].items() if v in mapping}
    tokenizer_json['model']['merges'] = remove_unused_merges(tokenizer_json['model']['merges'], tokenizer_json['model']['vocab'])
    
    # 추가된 토큰들 업데이트
    special_tokens_order = [t['id'] for t in tokenizer_json['added_tokens']]
    assert special_tokens_order==sorted(special_tokens_order)
    tokenizer_json['added_tokens'] = sorted([{**t, 'id': mapping[t['id']]} for t in tokenizer_json['added_tokens'] if t['id'] in mapping], key=lambda t: t['id'])
    
    # 후처리기 업데이트
    map_special_tokens(tokenizer_json.get('post_processor'), mapping)
    tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))
    return mapping, keep_indices

def shrink_model_embeddings(model, keep_indices, mapping):
    """
    모델의 임베딩 레이어를 축소하는 함수
    
    Args:
        model: 언어 모델
        keep_indices: 유지할 인덱스들
        mapping: 토큰 인덱스 매핑
    """
    import torch
    with torch.no_grad():
        # 유지할 토큰들만 선택
        row_select = torch.tensor(list(keep_indices))
        
        # 입력 임베딩 축소
        new_embed_t = torch.index_select(model.get_input_embeddings().weight.data, 0, row_select.to(model.get_input_embeddings().weight.data.device))
        # 출력 임베딩 축소
        new_lm_head = torch.index_select(model.get_output_embeddings().weight.data, 0, row_select.to(model.get_output_embeddings().weight.data.device))
        
        # 모델 크기 조정
        model.resize_token_embeddings(len(keep_indices))
        model.get_input_embeddings().weight.data[:] = new_embed_t
        model.get_output_embeddings().weight.data[:] = new_lm_head
        
        # 모델 설정의 특별 토큰 ID들 업데이트
        for config in [model.config, model.generation_config]:
            for k, v in list(config.to_dict().items()):
                if k.endswith('token_id'):
                    setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))

def shrink_embeddings(model, tokenizer, corpus=None, keep_token_ids=[], keep_tokens=[], remove_token_ids=[], keep_model_tokens=True, keep_special_tokens=True, keep_normalizer=False, keep_token_order=True):
    """
    모델과 토크나이저의 임베딩을 축소하는 메인 함수
    
    Args:
        model: 언어 모델
        tokenizer: 토크나이저
        corpus: 분석할 코퍼스 (선택사항)
        keep_token_ids: 유지할 토큰 ID 리스트
        keep_tokens: 유지할 토큰 문자열 리스트
        remove_token_ids: 제거할 토큰 ID 리스트
        keep_model_tokens: 모델 토큰 유지 여부
        keep_special_tokens: 특별 토큰 유지 여부
        keep_normalizer: 정규화기 유지 여부
        keep_token_order: 토큰 순서 유지 여부
    
    Returns:
        토큰 인덱스 매핑 딕셔너리
    """
    if not keep_normalizer: remove_tokenizer_normalizer(tokenizer)
    from collections import OrderedDict  # 순서가 있는 집합으로 사용
    keep_indices = OrderedDict()
    
    # 유지할 토큰들 수집
    keep_indices.update({k: None for k in keep_token_ids})
    keep_indices.update({tokenizer.vocab[t]: None for t in keep_tokens})
    if corpus is not None: keep_indices.update({k: None for k in tokenizer(corpus)['input_ids']})
    
    # 모델에서 사용되는 토큰들 유지
    if keep_model_tokens:
        for config in [model.config, model.generation_config]:
            for k, v in config.to_dict().items():
                if k.endswith('token_id'):
                    keep_indices.update({k: None for k in (v if isinstance(v, list) else [v])})
    
    # None 값과 제거할 토큰들 정리
    keep_indices.pop(None, None)
    for idx in remove_token_ids: keep_indices.pop(idx, None)
    
    # 토크나이저와 모델 축소 실행
    mapping, keep_indices = shrink_tokenizer_vocab(tokenizer, keep_indices, keep_special_tokens, keep_token_order)
    shrink_model_embeddings(model, keep_indices, mapping=mapping)
    return mapping

def fix_dtypes(model, fix_weights=True, fix_quant_states=True):
    """
    모델의 데이터 타입을 수정하는 함수
    
    Args:
        model: 언어 모델
        fix_weights: 가중치 타입 수정 여부
        fix_quant_states: 양자화 상태 타입 수정 여부
    
    Returns:
        수정된 모델
    """
    import torch
    for module in model.modules():
        weight = getattr(module, 'weight', None)
        if weight is not None:
            if torch.is_floating_point(weight):
                # 부동소수점 가중치 타입 수정
                if fix_weights and weight.dtype!=model.dtype:
                    module.to(model.dtype)
            else:
                # 양자화된 가중치의 상태 타입 수정
                qs = getattr(weight, 'quant_state', None)
                if qs is not None:
                    if fix_quant_states and qs.dtype!=model.dtype:
                        qs.dtype = model.dtype
    return model

def merge_peft_into_base(model):
    """
    PEFT(Parameter Efficient Fine-Tuning) 모델을 베이스 모델에 병합하는 함수
    
    Args:
        model: PEFT 모델
    
    Returns:
        병합된 베이스 모델
    """
    print('*** PEFT 모델을 베이스 모델에 병합 중...')
    assert is_peft_model(model)
    return fix_dtypes(model.merge_and_unload())

def save_model(store_path, model=None, tokenizer=None, merge=False):
    """
    모델과 토크나이저를 저장하는 함수
    
    Args:
        store_path: 저장 경로
        model: 저장할 모델 (선택사항)
        tokenizer: 저장할 토크나이저 (선택사항)
        merge: PEFT 모델 병합 여부
    
    Returns:
        처리된 모델
    """
    if merge: model = merge_peft_into_base(model)
    if store_path is not None:
        assert model is not None or tokenizer is not None
        print(f"*** {'병합된 ' if merge else ''}모델/토크나이저를 '{store_path}'에 저장 중...")
        if model is not None: model.save_pretrained(store_path)
        if tokenizer is not None:
            tokenizer.save_pretrained(store_path)
            # 불필요한 tokenizer.model 파일 삭제
            to_delete = os.path.join(store_path, 'tokenizer.model')
            if os.path.isfile(to_delete): os.remove(to_delete)
    return model

def is_unsloth_model(model):
    """Unsloth 모델인지 확인하는 함수"""
    return model.model_tags is not None and 'unsloth' in model.model_tags

def is_peft_model(model):
    """PEFT 모델인지 확인하는 함수"""
    return hasattr(model, 'peft_type')

def download_model(repo_id, store_path, get_name=lambda n: os.path.join(n.replace('/', '--'), 'transformers', 'default', '1')):
    """
    HuggingFace에서 모델을 다운로드하는 함수
    
    Args:
        repo_id: HuggingFace 모델 저장소 ID
        store_path: 로컬 저장 경로
        get_name: 파일명 생성 함수
    
    Returns:
        모델 경로
    """
    import os
    if os.path.exists(repo_id): return repo_id
    model_path = os.path.join(store_path, get_name(repo_id))
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download
        download_path = snapshot_download(repo_id=repo_id)
        os.makedirs(os.path.split(model_path)[0], exist_ok=True)
        os.symlink(download_path, model_path, target_is_directory=True)
    return model_path

def get_and_fix_peft_weights(store):
    """
    PEFT 가중치를 로드하고 수정하는 함수
    
    Args:
        store: PEFT 가중치 저장 경로
    
    Returns:
        수정된 state_dict
    """
    print(f"*** '{store}'에서 PEFT state_dict 로드 중...")
    from peft import load_peft_weights
    state_dict = load_peft_weights(store)
    # 불필요한 modules_to_save 관련 키들 제거
    for k in list(state_dict.keys()):
        if 'modules_to_save' in k:
            del state_dict[k]
            original_module_key = k.replace('.modules_to_save.', '.original_module.')
            if original_module_key in state_dict: del state_dict[original_module_key]
            assert k.replace('.modules_to_save.', '.') in state_dict
    return state_dict

def set_peft_weights(model, state_dict):
    """
    모델에 PEFT 가중치를 설정하는 함수
    
    Args:
        model: 타겟 모델
        state_dict: PEFT 가중치 딕셔너리
    """
    print(f"*** 모델 state_dict 설정 중...")
    from peft import set_peft_model_state_dict
    res = set_peft_model_state_dict(model, state_dict)
    assert not res.unexpected_keys

def load_peft_state(model, store):
    """
    저장된 PEFT 상태를 모델에 로드하는 함수
    
    Args:
        model: 타겟 모델
        store: PEFT 상태 저장 경로
    """
    set_peft_weights(model, get_and_fix_peft_weights(store))

def prepare_model(model, mode, tokenizer=None, formatter=None, shrink_embedding=False, dequantize=False, peft=[], local_files_only=False, add_special_tokens={}, set_pad_token=None, keep_tokens=[], keep_normalizer=None, peft_trainable=True, device_map=None, tf_grad_cp=True, tf_use_fa2=True, **kwargs):
    """
    모델과 토크나이저를 준비하는 메인 함수
    
    Args:
        model: 모델 경로 또는 모델 객체
        mode: 로드 모드 ('unsloth_4bit', 'transformers', 'transformers_bf16', 등)
        tokenizer: 토크나이저 (선택사항)
        formatter: 데이터 포맷터 (선택사항)
        shrink_embedding: 임베딩 축소 여부
        dequantize: 역양자화 여부
        peft: PEFT 설정 리스트
        local_files_only: 로컬 파일만 사용 여부
        add_special_tokens: 추가할 특별 토큰들
        set_pad_token: 패딩 토큰 설정
        keep_tokens: 유지할 토큰들
        keep_normalizer: 정규화기 유지 여부
        peft_trainable: PEFT 훈련 가능 여부
        device_map: 디바이스 매핑
        tf_grad_cp: 그래디언트 체크포인팅 사용 여부
        tf_use_fa2: Flash Attention 2 사용 여부
    
    Returns:
        (모델, 토크나이저, 포맷터) 튜플
    """
    if isinstance(model, str):
        assert tokenizer is None
        print(f"*** '{model}'에서 베이스 모델과 토크나이저 로드 중...")
        
        if mode=='unsloth_4bit':
            assert device_map is None, '지원되지 않음'
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(model_name=model, dtype=None, load_in_4bit=True, local_files_only=local_files_only, **kwargs)
        
        elif mode in ['transformers', 'transformers_bf16', 'transformers_4bit', 'transformers_bf16_4bit', 'tokenizer_only']:
            import torch
            model_load_args = {}
            if device_map is not None: model_load_args['device_map'] = device_map
            if tf_use_fa2: model_load_args['attn_implementation'] = 'flash_attention_2'
            if mode in ['transformers_bf16', 'transformers_bf16_4bit']: model_load_args['torch_dtype'] = torch.bfloat16
            elif mode in ['transformers_4bit', 'transformers_bf16_4bit']:
                from transformers import BitsAndBytesConfig
                nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
                model_load_args['quantization_config'] = nf4_config
            
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=local_files_only, **kwargs)
            model = AutoModelForCausalLM.from_pretrained(model, **model_load_args) if mode!='tokenizer_only' else None
            if tf_grad_cp and model is not None: model.gradient_checkpointing_enable()
        else: 
            raise NotImplementedError('알 수 없는 모드입니다.')
    
    # 특별 토큰 추가
    if add_special_tokens: tokenizer.add_special_tokens(add_special_tokens)
    if set_pad_token is not None: tokenizer.pad_token = set_pad_token
    
    # 포맷터 초기화
    if formatter is not None and not hasattr(formatter, 'corpus'):
        formatter = formatter(tokenizer=tokenizer)
    
    # 임베딩 축소
    if (shrink_embedding<len(tokenizer.vocab) if type(shrink_embedding)==int else shrink_embedding) or keep_normalizer is False:
        print('*** 임베딩 축소 중...')
        embedding_size_before_shrink = len(tokenizer.vocab)
        mapping = shrink_embeddings(model, tokenizer, formatter.get_corpus(), keep_tokens=keep_tokens, keep_normalizer=keep_normalizer)
        print(f'*** -> 임베딩 크기를 {embedding_size_before_shrink}에서 {len(mapping)} 단어로 축소했습니다.')
    
    # 역양자화
    if dequantize:
        print(f'*** 모델 역양자화 중...')
        model = model.dequantize()
    
    # PEFT 설정
    if len(peft):
        peft_trained = True if is_peft_model(model) else None
        for i, m in enumerate(peft):
            if peft_trained is True: model, peft_trained = merge_peft_into_base(model), None
            if isinstance(m, str):
                if peft_trained is False:
                    _, peft_trained = load_peft_state(model, m), True
                else:
                    print(f"*** '{m}'에서 PEFT 모델 로드 중...")
                    # unsloth 사용 시 주의: PeftModel로 로드하면 unsloth 최적화가 적용되지 않음
                    from peft import PeftModel
                    model, peft_trained = PeftModel.from_pretrained(model, m, trainable=peft_trainable), True
            else:
                assert peft_trained is None
                if isinstance(m, dict):
                    print('*** 새 PEFT 모델 생성 중...')
                    if is_unsloth_model(model):
                        from unsloth import FastLanguageModel
                        my_get_peft_model = FastLanguageModel.get_peft_model
                    else:
                        from peft import LoraConfig, get_peft_model
                        my_get_peft_model = lambda model, **kwargs: get_peft_model(model, LoraConfig(**kwargs))
                    model, peft_trained = my_get_peft_model(model, **m), False
                else: assert m is None
    
    return model, tokenizer, formatter

def training_run(model, formatter, dataset, train_args, max_seq_length, merge=False, store=None, packing=False, grad_acc_fix=False, optimizers=None):
    """
    모델 훈련을 실행하는 함수
    
    Args:
        model: 훈련할 모델
        formatter: 데이터 포맷터
        dataset: 훈련 데이터셋
        train_args: 훈련 인자들
        max_seq_length: 최대 시퀀스 길이
        merge: 훈련 후 병합 여부
        store: 모델 저장 경로
        packing: 시퀀스 패킹 사용 여부
        grad_acc_fix: 그래디언트 누적 수정 여부
        optimizers: 최적화 도구
    
    Returns:
        (모델, 훈련 통계) 튜플
    """
    assert merge is False, "훈련 후 병합은 작동하지 않는 것으로 보임 (적어도 unsloth에서는 저장된 병합 모델이 훈련되지 않은 가중치를 포함함!)"
    import torch
    from datasets import Dataset
    add_train_args = {}
    
    # Unsloth 또는 일반 Transformers 설정
    if is_unsloth_model(model):
        from unsloth import FastLanguageModel
        from unsloth import UnslothTrainer as Trainer
        from unsloth import UnslothTrainingArguments as TrainingArguments
        from unsloth import is_bfloat16_supported
        FastLanguageModel.for_training(model)
        add_train_args.update(fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported())
    else:
        from trl import SFTConfig as TrainingArguments
        from trl import SFTTrainer as Trainer
        model.train()
        add_train_args.update(bf16=True)

    # 토크나이저 설정
    formatter.tokenizer.padding_side = 'right'
    
    # Unsloth 모델의 임베딩을 float32로 변환
    if is_unsloth_model(model):
        for convert_to_float in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if convert_to_float.weight.dtype!=torch.float32: convert_to_float.to(torch.float32)

    add_args = {}
    if optimizers is not None: add_args['optimizers'] = optimizers

    # 트레이너 설정
    trainer = Trainer(
        model=model,
        tokenizer=formatter.tokenizer,
        data_collator=formatter.get_data_collator(),
        train_dataset=Dataset.from_list(dataset.as_list(formatter)),
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=None,
        packing=packing,  # 짧은 시퀀스에 대해 훈련 속도를 5배 향상시킬 수 있음
        **add_args,
        args=TrainingArguments(
            **add_train_args,
            **train_args
        ),
    )

    print('*** 훈련 실행 시작...')
    # 그래디언트 누적 수정 사용 여부에 따른 훈련 실행
    if grad_acc_fix and is_unsloth_model(model):
        from unsloth import unsloth_train
        trainer_stats = unsloth_train(trainer)
    else:
        if is_unsloth_model(model) and train_args['gradient_accumulation_steps']>1: 
            print('*** 경고: 결함이 있는 unsloth 그래디언트 누적을 사용 중')
        trainer_stats = trainer.train()
    
    try: print(f'*** -> 훈련이 {trainer_stats.metrics["train_runtime"]}초 소요되었습니다.')
    except: pass
    
    if store is not None: save_model(store, model, formatter.tokenizer, merge=merge)
    return model, trainer_stats

def inference_load(store, keys=True, result_dict=None, always_read_from_file=False):
    """
    저장된 추론 결과를 로드하는 함수
    
    Args:
        store: 결과 저장 경로
        keys: 로드할 키들 (True면 모든 키)
        result_dict: 결과를 저장할 딕셔너리
        always_read_from_file: 항상 파일에서 읽을지 여부
    
    Returns:
        로드된 결과 딕셔너리
    """
    if result_dict is None: result_dict = {}
    if store is not None:
        if keys is True: keys = os.listdir(store)
        for key in keys:
            if always_read_from_file or key not in result_dict:
                try:
                    with bz2.BZ2File(os.path.join(store, key)) as f: 
                        result_dict[key] = pickle.load(f)
                except: continue
    return result_dict

def inference_save(store, key, outputs):
    """
    추론 결과를 저장하는 함수
    
    Args:
        store: 저장 경로
        key: 결과 키
        outputs: 저장할 출력 결과
    """
    if store is not None:
        os.makedirs(store, exist_ok=True)
        with bz2.BZ2File(os.path.join(store, key), 'w') as f: 
            pickle.dump(outputs, f)

class Decoder(object):
    """
    모델 추론 결과를 디코딩하고 평가하는 클래스
    """
    def __init__(self, formatter, dataset, n_guesses, max_outputs=None, frac_score=False, quiet=False, name='', additional_decoders=None, prob_baseline=None):
        """
        Decoder 초기화
        
        Args:
            formatter: 데이터 포맷터
            dataset: 평가 데이터셋
            n_guesses: 허용되는 최대 추측 횟수
            max_outputs: 최대 출력 개수
            frac_score: 분수 점수 사용 여부
            quiet: 조용한 모드 여부
            name: 디코더 이름
            additional_decoders: 추가 디코더들
            prob_baseline: 확률 기준선
        """
        self.formatter = formatter
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}  # 디코딩된 결과들
        self.correct_solutions = {}  # 정답 솔루션들
        self.keys_lim = set()  # 제한된 추측 내에서 정답을 맞춘 키들
        self.keys_all = set()  # 모든 추측에서 정답을 맞춘 키들
        self.mult_cnt = {}  # 배수 카운트
        self.keys_cnt = {}  # 키 카운트
        self.frac_score = frac_score
        self.max_outputs = max_outputs
        self.quiet = quiet
        # 입력과 응답의 길이 정보 계산
        self.input_len = [{} if formatter is not None and formatter.tokenizer is None else ds.get_lengths(formatter, name='input') for ds in [dataset, dataset.mod(np.transpose, keep_key=True)]]
        self.reply_len = [{} if formatter is not None and formatter.tokenizer is None else ds.get_lengths(formatter, name='reply') for ds in [dataset, dataset.mod(np.transpose, keep_key=True)]]
        self.additional_decoders = additional_decoders
        self.name = name
        self.prob_tracker = {}  # 확률 추적기
        self.prob_tracker_best = {}  # 최고 확률 추적기
        self.prob_baseline = prob_baseline

    def score(self, *to_score):
        """
        점수를 계산하는 함수
        
        Args:
            *to_score: 점수를 매길 집합들
        
        Returns:
            (점수들, 총 개수) 튜플
        """
        scores = [(sum(1/self.mult_cnt[k.split('_')[0]] for k in s) if self.frac_score else len(s)) for s in to_score]
        score_cnt = len(self.mult_cnt if self.frac_score else self.keys_cnt)
        return scores, score_cnt

    def from_store(self, store, **kwargs):
        """
        저장소에서 결과를 로드하여 처리하는 함수
        
        Args:
            store: 저장소 경로
            **kwargs: 추가 인자들
        
        Returns:
            self 객체
        """
        for key, outputs in inference_load(store).items():
            self.process(key, outputs, **kwargs)
        return self

    def score_fmt(self, v):
        """점수 포맷팅 함수"""
        return f'{v:5.1f}' if self.frac_score else f'{v:3}'

    def process_single_output(self, key, output_len, decoded, print_func=print, len_info=None, device_info=None):
        """
        단일 출력을 처리하는 함수
        
        Args:
            key: 결과 키
            output_len: 출력 길이
            decoded: 디코딩된 결과
            print_func: 출력 함수
            len_info: 길이 정보
            device_info: 디바이스 정보
        """
        import numpy as np
        # 데이터셋 변환 역변환 적용
        inv_mod = {k: v if k.endswith('val') else self.dataset.invert_mod(v, key, inv_perm=(k.startswith('output') or k.startswith('score_all'))) for k, v in decoded.items()}
        base_key = key.split('.')[0]
        self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
        self.decoded_results[base_key][key] = inv_mod
        output = inv_mod.get('output')
        score = inv_mod.get('score')

        # 빠른 점수 계산
        self.keys_cnt[base_key] = self.keys_cnt.get(base_key, 0) + 1
        mult_key, mult_sub = (base_key.split('_') + ['0'])[:2]
        self.mult_cnt[mult_key] = max(self.mult_cnt.get(mult_key, 0), int(mult_sub) + 1)
        
        if len(self.dataset.replies):
            correct_solution = self.dataset.replies.get(base_key)
            if correct_solution is not None:
                correct_solution = correct_solution[0]
                self.correct_solutions[base_key] = correct_solution
                is_correct = correct_solution is not None and np.array_equal(correct_solution, output)
                if is_correct:
                    self.keys_all.add(base_key)
                    if self.keys_cnt[base_key] <= self.n_guesses: self.keys_lim.add(base_key)
            
            # 정답 여부 문자열 생성
            corr_str = 'cant_decode' if output is None else 'sol_unknown' if correct_solution is None else 'ALL_CORRECT' if is_correct else 'bad_xy_size' if np.shape(correct_solution)!=np.shape(output) else 'bad_content'
            (score_lim, score_all), score_cnt = self.score(self.keys_lim, self.keys_all)

            tp_arr = (key.count('transpose') + key.count('rot90')) % 2
            msc = None if score is None else np.sum(score)
            fsc = inv_mod.get('score_val')
            
            # 확률 추적
            if output is not None and fsc is not None:
                pt = self.prob_tracker[base_key] = self.prob_tracker.get(base_key, {})
                hash = tuple(map(tuple, output))
                prob = pt[hash] = pt.get(hash, 0) + (np.exp(fsc) if self.prob_baseline is None else fsc - np.log(self.prob_baseline))
                current_best = self.prob_tracker_best.get(base_key)
                if current_best is None or current_best[0]<prob:
                    self.prob_tracker_best[base_key] = (prob, output)
            
            # 결과 출력 포맷팅
            fmt_name = f'{self.name}: ' if self.name else ''
            msc_print = f'{min(-msc, 9.99999):7.5f}' if msc is not None else 'unknown'
            fsc_print = f'{min(-fsc, 9.99999):7.5f}' if fsc is not None else 'unknown'
            if not self.quiet: 
                print_func(f" {fmt_name}acc: {self.score_fmt(score_lim)}/{score_cnt:3}={min(score_lim/score_cnt, 0.999):5.1%} (2-guess), {self.score_fmt(score_all)}/{score_cnt:3}={min(score_all/score_cnt, 0.999):5.1%} (any);{f' {device_info}' if device_info else ''} tok:{self.input_len[tp_arr].get(base_key, '?'):>4}+{self.reply_len[tp_arr].get(base_key, '?'):>3}>{'n/a' if output_len is None else output_len:>3} {corr_str}:{msc_print}|{fsc_print} [{key}]")

    def get_current_best(self, base_key):
        """
        현재 최고 결과를 가져오는 함수
        
        Args:
            base_key: 기본 키
        
        Returns:
            최고 결과 또는 None
        """
        current_best = self.prob_tracker_best.get(base_key)
        return None if current_best is None else current_best[1]

    def process_single_decode(self, key, de_tokenized, print_func=print, **kwargs):
        """
        단일 디코딩 결과를 처리하는 함수
        
        Args:
            key: 결과 키
            de_tokenized: 디토크나이즈된 결과
            print_func: 출력 함수
            **kwargs: 추가 인자들
        """
        # 호환성을 위한 포맷 확인
        if len(de_tokenized)==3 and not isinstance(de_tokenized[1], float):  
            output_len, *data = de_tokenized
            score_val = None
        else: 
            output_len, score_val, *data = de_tokenized
        
        if self.formatter is None:
            assert len(data) == 1
            decoded = [data[0]]
        else: 
            decoded = self.formatter.decode_to_array(*data)
        
        # 점수 값 추가
        for d in decoded: d['score_val'] = score_val
        
        # 각 디코딩 결과 처리
        for i, dec in enumerate(decoded):
            if i==0: 
                self.process_single_output(key, output_len, dec, print_func=print_func, **kwargs)
            elif self.additional_decoders:
                if i-1<len(self.additional_decoders): 
                    self.additional_decoders[i-1].process_single_output(key, output_len, dec, print_func=print_func, **kwargs)
                else: 
                    print_func(f'{key} 출력 #{i}에 사용할 수 있는 디코더가 없습니다')
            else: 
                self.process_single_output(f'{key}.fix{i}', output_len, dec, print_func=print_func, **kwargs)

    def process(self, key, de_tokenized, **kwargs):
        """
        디토크나이즈된 결과들을 처리하는 함수
        
        Args:
            key: 결과 키
            de_tokenized: 디토크나이즈된 결과들
            **kwargs: 추가 인자들
        """
        for i, d in enumerate(de_tokenized):
            if self.max_outputs is None or i<=self.max_outputs:
                self.process_single_decode(f'{key}.out{i}', d, **kwargs)

    def get_unsolved_keys(self):
        """
        아직 해결되지 않은 키들을 반환하는 함수
        
        Returns:
            해결되지 않은 키들의 리스트
        """
        unsolved = []
        for base_key, reply in self.dataset.replies.items():
            if not any(np.array_equal(reply[0], s.get('output')) for s in self.decoded_results.get(base_key, {}).values()):
                unsolved.append(base_key)
        return unsolved

    def run_selection_algo(self, selection_algorithm):
        """
        선택 알고리즘을 실행하는 함수
        
        Args:
            selection_algorithm: 선택 알고리즘 함수
        
        Returns:
            선택된 결과들의 딕셔너리
        """
        return {bk: (selection_algorithm({k: g for k, g in v.items() if g.get('output') is not None}) if any(g.get('output') is not None for g in v.values()) else []) for bk, v in self.decoded_results.items()}

    def benchmark_selection_algos(self, selection_algorithms, skip_failed=True):
        """
        선택 알고리즘들을 벤치마크하는 함수
        
        Args:
            selection_algorithms: 테스트할 선택 알고리즘들
            skip_failed: 실패한 알고리즘 건너뛸지 여부
        
        Returns:
            벤치마크 결과 딕셔너리
        """
        import numpy as np
        results = {}
        print('*** 선택 알고리즘 벤치마크 중...')
        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            try:
                selected = self.run_selection_algo(selection_algorithm)
                if self.formatter is not None:
                    for sols in selected.values():
                        for s in sols:
                            assert self.formatter.is_valid_solution(s), f'유효하지 않은 솔루션 발견 {s}'
                correct_keys = {k for k, v in selected.items() if self.correct_solutions.get(k) is not None and any(np.array_equal(guess, self.correct_solutions[k]) for guess in v[:self.n_guesses])}
                (score,), score_cnt = self.score(correct_keys)
                results[name] = score
                print(f" acc: {score:5.1f}/{score_cnt:3}={score/score_cnt:6.2%} ('{name}')")
            except:
                print(f" {'실행 실패':>21} ('{name}')")
                if not skip_failed: raise
        return results

    def calc_augmented_scores(self, model, base_keys=None, store=None, seed=0, max_len=None, make_unique=True, quiet=False, **kwargs):
        """
        증강된 점수를 계산하는 함수
        
        Args:
            model: 평가할 모델
            base_keys: 기본 키들 (None이면 모든 키)
            store: 저장 경로
            seed: 랜덤 시드
            max_len: 최대 길이
            make_unique: 고유성 확보 여부
            quiet: 조용한 모드 여부
            **kwargs: 추가 인자들
        """
        if base_keys is None: base_keys = list(self.decoded_results.keys())
        if store is not None: store = f'{store}_new'  # 새 포맷은 하위 호환되지 않으므로 새 폴더 사용
        
        for bk in (base_keys if quiet else tqdm(base_keys, desc='증강된 점수 계산', file=sys.stdout)):
            res = self.decoded_results.get(bk, {})
            known_scores = {}
            for k, v in sorted(res.items()):
                if 'output' in v:
                    k_store = None if store is None else os.path.join(store, k)
                    id = tuple(map(tuple, v['output']))
                    if not (make_unique and id in known_scores):
                        try:
                            assert k_store is not None
                            with bz2.BZ2File(k_store) as f: 
                                known_scores[id] = pickle.load(f)
                            # 하위 호환성을 위한 포맷 변환
                            if isinstance(known_scores[id], list): 
                                known_scores[id] = dict(score_multi=known_scores[id])  
                            k_store = None
                        except:
                            # 임시 데이터셋 생성하여 점수 계산
                            temp_dataset = self.dataset.__class__(
                                keys=[bk],
                                queries={bk: self.dataset.queries.get(bk)},
                                replies={bk: [v['output'].tolist()]},
                            )
                            temp_decoder = self.__class__(self.formatter, temp_dataset, n_guesses=self.n_guesses, quiet=True)
                            temp_dataset = temp_dataset.augment(**kwargs, seed=(seed+hash(k)+hash(id)) % 1024**2, quiet=True)
                            if max_len is not None: 
                                temp_dataset = temp_dataset.cut_to_len(formatter=self.formatter, name='input', max_len=max_len, quiet=True)
                            for x in temp_dataset.as_list(self.formatter): 
                                calc_score(**x, formatter=self.formatter, model=model, decoder=temp_decoder)
                            
                            # 다양한 점수 메트릭 저장
                            known_scores[id] = dict(
                                score_multi=[np.sum(x['score']) for x in temp_decoder.decoded_results[bk].values()],
                                score_multi_nl=[x['score_val'] for x in temp_decoder.decoded_results[bk].values()],
                                score_multi_array=np.array([x['score'] for x in temp_decoder.decoded_results[bk].values()]),
                                score_multi_array_cum=np.array([x['score_cum'] for x in temp_decoder.decoded_results[bk].values()]),
                                score_multi_array_all=np.array([x['score_all'] for x in temp_decoder.decoded_results[bk].values()]),
                                score_multi_array_all_cum=np.array([x['score_all_cum'] for x in temp_decoder.decoded_results[bk].values()]),
                            )
                            if k_store is not None:
                                os.makedirs(store, exist_ok=True)
                                with bz2.BZ2File(k_store, 'w') as f: 
                                    pickle.dump(known_scores[id], f)
                    v.update(known_scores[id])

def turbo_dfs(model, logits, path, eos_token_id, max_new_tokens, max_score, max_score_greedy, temperature, suppress_tokens, torch, score=0.0, pos=0, cache=None):
    """
    터보 깊이 우선 탐색 함수 (효율적인 빔 서치 대안)
    
    Args:
        model: 언어 모델
        logits: 로짓 값들
        path: 미리 계산된 경로
        eos_token_id: 문장 종료 토큰 ID
        max_new_tokens: 최대 새 토큰 수
        max_score: 최대 점수
        max_score_greedy: 탐욕적 최대 점수
        temperature: 온도 파라미터
        suppress_tokens: 억제할 토큰들
        torch: PyTorch 모듈
        score: 현재 점수
        pos: 현재 위치
        cache: 캐시
    
    Returns:
        (점수, 접미사, 로짓들) 튜플들의 리스트
    """
    logits, next_logits = logits[0], (logits[1:] if len(logits)>1 else None)
    nll = -(logits / temperature).detach().float().log_softmax(-1).cpu().numpy()
    greedy_index = nll.argmin(-1).item()
    nll = list(enumerate(nll))
    
    # 미리 계산된 경로가 있으면 먼저 따라가기
    if path: nll[0], nll[path[0]], path = nll[path[0]], nll[0], path[1:]  
    
    suffixes = []
    for i, s in nll:
        next_score = score + s
        allowed_max_score = max_score_greedy if i==greedy_index else max_score
        if next_score < allowed_max_score:
            if i==eos_token_id: 
                next_suffixes = [(next_score, [], [])]
            elif max_new_tokens>1:
                if next_logits is None:
                    # 캐시 크기 조정
                    if pos<cache[0][0][0].shape[2]: 
                        cache[0] = tuple(tuple(c[:, :, :pos] for c in l) for l in cache[0])
                    # 다음 토큰 생성
                    next_logits, cache[0] = model(
                        input_ids= torch.full((1,1), i, device=model.device),
                        position_ids=torch.full((1,1), pos, device=model.device),
                        past_key_values=cache[0],
                    )[:2]
                    next_logits = next_logits[0]  # 배치 차원 제거
                # 재귀 호출
                next_suffixes = turbo_dfs(model, logits=next_logits, path=path, eos_token_id=eos_token_id, max_new_tokens=max_new_tokens-1, max_score=max_score, max_score_greedy=allowed_max_score, temperature=temperature, suppress_tokens=suppress_tokens, torch=torch, score=next_score, pos=pos+1, cache=cache)
            else: 
                next_suffixes = []
            
            # 접미사에 현재 토큰과 로짓 추가
            for suffix in next_suffixes:
                suffix[1].append(i)
                suffix[2].append(logits)
            suffixes.extend(next_suffixes)
        next_logits = None
    return suffixes

def inference_turbo_dfs(model, input_ids, eos_token_id, max_new_tokens, min_prob, min_prob_greedy=1, temperature=1.0, suppress_tokens=[], path=[], attention_mask=None):
    """
    터보 DFS를 사용한 추론 함수
    
    Args:
        model: 언어 모델
        input_ids: 입력 토큰 ID들
        eos_token_id: 문장 종료 토큰 ID
        max_new_tokens: 최대 새 토큰 수
        min_prob: 최소 확률
        min_prob_greedy: 탐욕적 최소 확률
        temperature: 온도 파라미터
        suppress_tokens: 억제할 토큰들
        path: 경로
        attention_mask: 어텐션 마스크
    
    Returns:
        정렬된 결과 리스트 (점수, 접미사, 점수 배열)
    """
    import torch
    with torch.no_grad():
        assert attention_mask is None or attention_mask.all(), '구현되지 않음'
        input_ids = torch.as_tensor(input_ids, device=model.device, dtype=int)
        if input_ids.ndim==2: input_ids = input_ids.squeeze(0)
        assert input_ids.ndim==1, '배칭은 지원되지 않음'
        
        # 점수 임계값 계산
        max_score = -np.log(min_prob)
        max_score_greedy = (-np.log(min_prob_greedy)) if min_prob_greedy>0 else float('inf')  
        max_score_greedy = max(max_score, max_score_greedy)
        
        if path is None: path = []
        if len(path) and path[-1]==eos_token_id: path = path[:-1]
        
        with torch.no_grad():
            full_path = input_ids
            if len(path): 
                full_path = torch.cat([full_path, torch.as_tensor(path, device=model.device)])
            logits, cache = model(input_ids=full_path[np.newaxis])[:2]
            logits = logits[0, len(input_ids)-1:]
        
        # 터보 DFS 실행
        result = turbo_dfs(model, logits=logits, path=path, eos_token_id=eos_token_id, max_new_tokens=max_new_tokens, max_score=max_score, max_score_greedy=max_score_greedy, temperature=temperature, suppress_tokens=suppress_tokens, torch=torch, score=0.0, pos=len(input_ids), cache=[cache])
        
        # 결과 정렬하여 반환
        return sorted([(score_val, np.array(suffix[::-1]), torch.stack(score_arr[::-1]).float().cpu().numpy()) for score_val, suffix, score_arr in result], key=lambda x:x[0])

def inference_step(tokenized, model, remove_token_type_ids=True, num_beams=1, formatter=None, min_prob=None, current_best=None, **kwargs):
    """
    추론 단계를 실행하는 함수
    
    Args:
        tokenized: 토크나이즈된 입력
        model: 언어 모델
        remove_token_type_ids: 토큰 타입 ID 제거 여부
        num_beams: 빔 개수
        formatter: 포맷터
        min_prob: 최소 확률
        current_best: 현재 최고 결과
        **kwargs: 추가 인자들
    
    Returns:
        (토큰 출력, 점수 출력) 튜플
    """
    import torch
    if remove_token_type_ids: tokenized.pop('token_type_ids', None)
    
    if min_prob is not None:
        assert num_beams==1
        # 터보 DFS 사용
        gen = inference_turbo_dfs(model, **tokenized.to(model.device), path=current_best, min_prob=min_prob, eos_token_id=formatter.tokenizer.eos_token_id, **kwargs)
        tokens_out = [[g[1] for g in gen]]
        scores_out = [[g[2] for g in gen]]
    elif is_unsloth_model(model) and num_beams > 1:
        assert False, 'unsloth는 빔 서치를 지원하지 않습니다'
    else:
        # 표준 생성 방식
        gen = model.generate(**tokenized.to(model.device), return_dict_in_generate=True, output_logits=True, use_cache=True, **kwargs)
        tokens_out = gen['sequences'][:, torch.newaxis, tokenized['input_ids'].shape[-1]:].cpu().numpy().copy()
        scores_out = torch.stack(gen['logits'], axis=-2)[:, torch.newaxis].float().cpu().numpy().copy()
    return tokens_out, scores_out

def process_inference_output(key, outputs, formatter, store=None, decoder=None, decoder_args={}):
    """
    추론 출력을 처리하는 함수
    
    Args:
        key: 결과 키
        outputs: 출력 결과들
        formatter: 포맷터
        store: 저장 경로
        decoder: 디코더
        decoder_args: 디코더 인자들
    
    Returns:
        디토크나이즈된 결과
    """
    de_tokenized = [formatter.de_tokenize(*output) for output in zip(*outputs)]
    inference_save(store, key, de_tokenized)
    if decoder is not None: decoder.process(key, de_tokenized, **decoder_args)
    return de_tokenized

def inference_run_v2(model, formatter, dataset, decoder=None, max_new_tokens=None, max_batch_size=1, store=None, result_dict=None, rerun_empty=False, retrain=None, use_turbo=False, group_multi_output=True, **kwargs):
    """
    추론 실행 함수 (버전 2)
    
    Args:
        model: 언어 모델
        formatter: 데이터 포맷터
        dataset: 데이터셋
        decoder: 디코더
        max_new_tokens: 최대 새 토큰 수
        max_batch_size: 최대 배치 크기
        store: 저장 경로
        result_dict: 결과 딕셔너리
        rerun_empty: 빈 결과 재실행 여부
        retrain: 재훈련 함수
        use_turbo: 터보 모드 사용 여부
        group_multi_output: 다중 출력 그룹화 여부
        **kwargs: 추가 인자들
    
    Returns:
        결과 딕셔너리
    """
    import torch
    assert max_batch_size==1, '지원되지 않음'

    with torch.no_grad():
        print('*** 저장된 데이터 로드 중...')
        if result_dict is None: result_dict = {}
        result_dict = inference_load(store, dataset.keys, result_dict)
        
        # 키들을 기본 키별로 그룹화
        by_base_key = {}
        needs_rerun = {}
        base_key_list = []
        for key in dataset.keys:
            base_key = key.split('.')[0]
            if group_multi_output: base_key = base_key.split('_')[0]
            if base_key not in by_base_key: base_key_list.append(base_key)
            bk_list = by_base_key[base_key] = by_base_key.get(base_key, [])
            bk_list.append(key)
        
        # 재실행이 필요한 키들 찾기
        for base_key, keys in by_base_key.items():
            for key in keys:
                de_tokenized = result_dict.get(key)
                if de_tokenized is None or (rerun_empty and not de_tokenized):
                    bk_list = needs_rerun[base_key] = needs_rerun.get(base_key, [])
                    bk_list.append(key)
                elif decoder is not None: 
                    decoder.process(key, de_tokenized)

        # 모델을 추론 모드로 설정
        formatter.tokenizer.padding_side = 'left'
        if max_new_tokens is None: max_new_tokens = formatter.max_new_tokens()
        if is_unsloth_model(model):
            from unsloth import FastLanguageModel
            FastLanguageModel.for_inference(model)
        else: 
            model.eval()

        print('*** 추론 실행 시작...')
    try:
        with tqdm(base_key_list, file=sys.stdout) as pbar:
            for base_key in pbar:
                run_keys = needs_rerun.get(base_key)
                if run_keys:
                    # 재훈련이 필요한 경우
                    if retrain is not None:
                        retrain_dataset = dataset.keep_key_startswith(base_key)
                        print(f"키 '{base_key}'에 대해 모델 재훈련 중 (retrain_dataset_size={len(retrain_dataset.keys)})")
                        retrain(model, retrain_dataset)
                        if is_unsloth_model(model): FastLanguageModel.for_inference(model)
                    
                    with torch.no_grad():
                        for key in run_keys:
                            # 입력 텍스트 준비
                            input_text = dataset.get(key, formatter)['input']
                            batch = formatter.tokenizer([input_text], return_tensors='pt')
                            
                            # 터보 모드에서 현재 최고 결과 사용
                            current_best = decoder.get_current_best(key.split('.')[0]) if use_turbo else None
                            if current_best is not None:
                                current_best = dataset.forward_mod(current_best, key)
                                current_best = formatter.fmt_reply([current_best])
                                current_best = formatter.tokenizer(input_text+current_best)['input_ids'][batch['input_ids'].shape[-1]:]
                            
                            # 추론 실행
                            batch_out = inference_step(batch, model, formatter=formatter, max_new_tokens=max_new_tokens, current_best=current_best, **kwargs)
                            outputs = [x[0] for x in batch_out]
                            result_dict[key] = process_inference_output(key, outputs, formatter, store=store, decoder=decoder, decoder_args=dict(print_func=pbar.write))
        print('*** 추론 실행 완료.')
    except KeyboardInterrupt: 
        print('*** Ctrl+C 눌림, 추론 실행 중단.')
    return result_dict

class Retrainer(object):
    """
    모델 재훈련을 담당하는 클래스
    """
    def __init__(self, n, aug_opts, reload_state_dict=None, **kwargs):
        """
        Retrainer 초기화
        
        Args:
            n: 훈련 샘플 수
            aug_opts: 데이터 증강 옵션들
            reload_state_dict: 재로드할 state_dict
            **kwargs: 추가 인자들
        """
        self.n = n
        self.aug_opts = aug_opts
        self.reload_state_dict = reload_state_dict
        self.kwargs = kwargs

    def preprocess(self, dataset):
        """
        데이터셋을 전처리하는 함수
        
        Args:
            dataset: 입력 데이터셋
        
        Returns:
            전처리된 데이터셋
        """
        # 필요한 수만큼 데이터 증강
        ds = [dataset.augment(quiet=True, shfl_keys=True, **self.aug_opts) for _ in range((self.n-1)//dataset.length()+1)]
        ds = ds[0] if len(ds)==1 else ds[0].append(*ds[1:])
        ds, _ = ds.split_at_pos(self.n)
        return ds

    def __call__(self, model, dataset):
        """
        재훈련 실행
        
        Args:
            model: 재훈련할 모델
            dataset: 훈련 데이터셋
        """
        if self.reload_state_dict is not None: 
            set_peft_weights(model, self.reload_state_dict)
        
        assert is_unsloth_model(model), '구현되지 않음'
        if is_unsloth_model(model):
            from unsloth import FastLanguageModel
            FastLanguageModel.for_training(model)
        else: 
            model.train()
        
        training_run(model, dataset=self.preprocess(dataset), **self.kwargs)

def calc_score(key, input, reply, formatter, model, store=None, decoder=None, **_):
    """
    주어진 입력-응답 쌍에 대한 점수를 계산하는 함수
    
    Args:
        key: 데이터 키
        input: 입력 텍스트
        reply: 응답 텍스트
        formatter: 데이터 포맷터
        model: 언어 모델
        store: 저장 경로
        decoder: 디코더
        **_: 무시되는 추가 인자들
    """
    import torch
    with torch.no_grad():
        # 입력 길이 계산
        input_len = len(formatter.tokenizer(input)['input_ids'])
        # 전체 시퀀스 토크나이즈
        tokenized = formatter.tokenizer([input+reply], return_tensors='pt')
        # 응답 부분의 토큰들만 추출
        reply_tok = tokenized['input_ids'][0][input_len:].cpu().numpy().copy()
        # 응답 부분의 로그 확률 계산
        reply_log = model.forward(**tokenized.to(model.device))['logits'][0, input_len-1: -1].float().cpu().numpy().copy()
        # 결과 처리
        process_inference_output(key, (reply_tok[torch.newaxis], reply_log[torch.newaxis]), formatter, store=store, decoder=decoder)

def mem_info(gpu_id=0):
    """
    GPU 메모리 정보를 출력하는 함수
    
    Args:
        gpu_id: GPU ID (기본값: 0)
    """
    import torch
    try:
        gpu_stats = torch.cuda.get_device_properties(gpu_id)
        usage = torch.cuda.max_memory_reserved() / 1024**3
        avail = gpu_stats.total_memory / 1024**3
        print(f"*** GPU: {gpu_stats.name}, 사용량 {usage:.3} / {avail:.3} GB.")
    except: 
        print('*** 메모리 통계를 가져오는 중 예외가 발생했습니다.')