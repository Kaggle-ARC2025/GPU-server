# arc_loader.py
# 전체 내용은 원본 노트북의 %%writefile arc_loader.py 셀에서 복사해주세요.
# 이 파일은 ARC 데이터셋 로드 및 처리 관련 클래스와 함수들을 포함합니다.
import json
import numpy as np
import hashlib
import os, sys
import random
from tqdm import tqdm
from glob import glob
import itertools

def cut_at_token(output, token_id):
    """
    특정 토큰에서 출력을 자르는 함수
    
    Args:
        output: 출력 시퀀스
        token_id: 자를 토큰 ID
    
    Returns:
        자른 출력 시퀀스
    """
    eos_positions = (output==token_id).nonzero()[0]
    return output[:eos_positions[0]] if len(eos_positions) else output

def shuffled(data_list):
    """데이터 리스트를 무작위로 섞는 함수"""
    return np.random.permutation(data_list).tolist()

def permute_mod(a, descriptor, invert=False):
    """
    배열에 순열 변환을 적용하는 함수
    
    Args:
        a: 변환할 배열
        descriptor: 순열을 나타내는 문자열
        invert: 역변환 여부
    
    Returns:
        순열이 적용된 배열
    """
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation)==list(range(10))
    a = np.asarray(a)
    if a.ndim==3:
        # 3차원 배열의 경우 (색상 차원)
        if not invert: permutation = np.argsort(permutation)
        a = a[..., permutation]
    else:
        # 2차원 배열의 경우 (그리드)
        assert a.ndim==2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
    return a

def permute_rnd_col_(query):
    """배경색(0)을 유지하고 나머지 색상을 무작위로 순열하는 함수"""
    permutation = [0]+(1+np.random.permutation(9)).tolist()
    return 'permute' + ''.join(map(str, permutation))

def permute_rnd_all_(query):
    """모든 색상을 무작위로 순열하는 함수"""
    permutation = np.random.permutation(10).tolist()
    return 'permute' + ''.join(map(str, permutation))

def permute_cnt_col_(query):
    """
    배경색을 유지하고 나머지 색상을 빈도순으로 정렬하는 함수
    (무작위성을 동점자 해결 기준으로 사용)
    """
    elements, frequency = np.unique(np.concatenate([list(range(10))]+[np.array(x['input']).ravel() for x in query['train']]), return_counts=True)
    permutation = [0]+sorted(np.random.permutation(9)+1, key=lambda i: frequency[i], reverse=True)  
    return 'permute' + ''.join(map(str, permutation))

def permute_cnt_all_(query):
    """모든 색상을 빈도순으로 정렬하는 함수 (무작위성을 동점자 해결 기준으로 사용)"""
    elements, frequency = np.unique(np.concatenate([list(range(10))]+[np.array(x['input']).ravel() for x in query['train']]), return_counts=True)
    permutation = sorted(np.random.permutation(10), key=lambda i: frequency[i], reverse=True)  
    return 'permute' + ''.join(map(str, permutation))

# 다양한 순열 변환 옵션들
permute_rnd_col = (permute_mod, permute_rnd_col_)  # 배경색 유지, 무작위 순열
permute_rnd_all = (permute_mod, permute_rnd_all_)  # 전체 무작위 순열
permute_cnt_col = (permute_mod, permute_cnt_col_)  # 배경색 유지, 빈도순 정렬
permute_cnt_all = (permute_mod, permute_cnt_all_)  # 전체 빈도순 정렬
permute_None = (np.copy, None)  # 순열 없음 (복사만)

class ArcDataset(object):
    """ARC 데이터셋을 처리하는 메인 클래스"""
    
    @staticmethod
    def forward_mod(a, key, use_perm=True, is_output=True):
        """
        키에 따라 배열에 순방향 변환을 적용하는 함수
        
        Args:
            a: 변환할 배열
            key: 변환 정보가 포함된 키
            use_perm: 순열 사용 여부
            is_output: 출력 데이터 여부
        
        Returns:
            변환된 배열
        """
        if a is None: return a
        for op in key.split('.')[1:]:
            # 'I' 접두사는 입력에만 적용되는 변환을 의미
            if op.startswith('I'):
                if is_output: continue
                op = op[1:]
            # 다양한 변환 적용
            if   op=='rot90':              a = np.rot90(a)  # 90도 회전
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)  # 전치
            elif op.startswith('permute'): a = permute_mod(a, op, invert=False) if use_perm else a  # 순열
            elif op.startswith('copy'):    a = np.copy(a)  # 복사
            elif op.startswith('out'):     a = a  # 출력 표시
            elif op.startswith('ex'):      a = a  # 예제 표시
            elif op.startswith('fix'):     a = a  # 수정 표시
            elif op.startswith('ice'):     a = a  # icecuber 솔루션 추가용
            else: raise NotImplementedError(f"연산 '{op}'의 역변환을 알 수 없습니다.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True, is_output=True):
        """
        키에 따라 배열에 역방향 변환을 적용하는 함수
        
        Args:
            a: 변환할 배열
            key: 변환 정보가 포함된 키
            inv_perm: 순열 역변환 사용 여부
            is_output: 출력 데이터 여부
        
        Returns:
            역변환된 배열
        """
        if a is None: return a
        # 변환을 역순으로 적용
        for op in key.split('.')[1:][::-1]:
            if op.startswith('I'):
                if is_output: continue
                op = op[1:]
            # 역변환 적용
            if   op=='rot90':              a = np.rot90(np.rot90(np.rot90(a)))  # 270도 회전 (90도의 역)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)  # 전치 (자기 자신이 역변환)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('fix'):     a = a
            elif op.startswith('ice'):     a = a
            else: raise NotImplementedError(f"연산 '{op}'의 역변환을 알 수 없습니다.")
        return a

    def __init__(self, queries, replies={}, keys=None, is_orig=False, is_fake=False):
        """
        ArcDataset 초기화
        
        Args:
            queries: 문제 데이터 딕셔너리
            replies: 답안 데이터 딕셔너리
            keys: 사용할 키 리스트
            is_orig: 원본 데이터셋 여부
            is_fake: 가짜 테스트 세트 여부
        """
        if keys is not None: keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.is_fake = is_fake
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.faulty = {}  # 결함이 있는 데이터 추적
        self.transposed_dataset = None  # 전치된 데이터셋 캐시

    @classmethod
    def empty(cls):
        """빈 데이터셋을 생성하는 클래스 메서드"""
        return cls(queries={}, replies={}, keys=[])

    def change_keys(self, keys, keep_flags=False):
        """
        키를 변경하여 새로운 데이터셋을 생성하는 함수
        
        Args:
            keys: 새로운 키 리스트
            keep_flags: 플래그 유지 여부
        
        Returns:
            새로운 ArcDataset 인스턴스
        """
        flags = dict(is_fake=self.is_fake, is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, queries_file):
        """
        파일에서 문제 데이터를 로드하는 클래스 메서드
        
        Args:
            queries_file: 문제 파일 경로
        
        Returns:
            로드된 ArcDataset 인스턴스
        """
        print(f"*** '{queries_file}'에서 문제 로드 중...")
        with open(queries_file) as f: queries = f.read()
        # 가짜 테스트 세트 감지 (특정 MD5 해시로 판별)
        is_fake = hashlib.md5(queries.encode('utf-8')).hexdigest().lower()=='a6b7dac3cab03abf2eb333e16610d6dc'
        if is_fake: print("*** -> 가짜 테스트 세트 감지됨, 'is_fake' 플래그를 True로 설정.")
        return cls(
            queries=json.loads(queries),
            is_fake=is_fake,
            is_orig=True,
        )

    def load_replies(self, replies_file):
        """
        답안 파일을 로드하는 함수
        
        Args:
            replies_file: 답안 파일 경로
        
        Returns:
            self 객체
        """
        print(f"*** '{replies_file}'에서 솔루션 로드 중...")
        with open(replies_file) as f: replies = f.read()
        replies_parsed = json.loads(replies)
        self.replies = {k: replies_parsed[k] for k in self.keys}
        return self

    def split_multi_replies(self):
        """
        다중 테스트 케이스를 개별 키로 분할하는 함수
        
        Returns:
            분할된 새로운 ArcDataset
        """
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        return self.__class__(
            keys=[f'{k}_{i}' for k, i in key_indices],
            queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
            replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
        )

    def move_test_to_train(self):
        """
        테스트 데이터를 훈련 데이터로 이동하는 함수
        
        Returns:
            변환된 새로운 ArcDataset
        """
        new_queries = {k: {'train': self.queries[k]['train'] + [{**t, 'output': self.replies[k][i]} for i, t in enumerate(self.queries[k]['test'])], 'test': []} for k in self.keys}
        return self.__class__(queries=new_queries, keys=[k for k in self.keys])

    def last_train_ex_for_test(self):
        """
        마지막 훈련 예제를 테스트로 사용하는 함수
        
        Returns:
            변환된 새로운 ArcDataset
        """
        assert not self.replies
        new_queries = {k: {'train': self.queries[k]['train'][:-1], 'test': [{'input': self.queries[k]['train'][-1]['input']}]} for k in self.keys}
        new_replies = {k: [self.queries[k]['train'][-1]['output']] for k in self.keys}
        return self.__class__(queries=new_queries, replies=new_replies, keys=[k for k in self.keys])

    def length(self):
        """데이터셋의 길이를 반환하는 함수"""
        return len(self.keys)

    def shuffled(self, seed=None):
        """
        키를 무작위로 섞은 새로운 데이터셋을 반환하는 함수
        
        Args:
            seed: 랜덤 시드
        
        Returns:
            섞인 새로운 ArcDataset
        """
        if seed is not None: np.random.seed(seed)
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    def sorted(self, **kwargs):
        """키를 정렬한 새로운 데이터셋을 반환하는 함수"""
        return self.__class__(queries=self.queries, replies=self.replies, keys=sorted(self.keys, **kwargs))

    @staticmethod
    def append(*datasets):
        """
        여러 데이터셋을 결합하는 정적 메서드
        
        Args:
            *datasets: 결합할 데이터셋들
        
        Returns:
            결합된 새로운 ArcDataset
        """
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys   =[k    for d in datasets for k    in d.keys           ],
        )

    def sort_ex_by_input_size(self, seed=42, reverse=False):
        """
        예제를 입력 크기순으로 정렬하는 함수
        
        Args:
            seed: 랜덤 시드
            reverse: 역순 정렬 여부
        
        Returns:
            정렬된 새로운 ArcDataset
        """
        np.random.seed(seed)
        sort_key = lambda ex: np.prod(np.shape(ex['input']))
        new_queries = {k2: {k: (sorted(np.random.permutation(np.array(v, dtype=object)), key=sort_key, reverse=reverse) if k=='train' else v) for k, v in v2.items()} for k2, v2 in self.queries.items()}
        return self.__class__(queries=new_queries, replies=self.replies, keys=[k for k in self.keys])

    def interleave(self, block_size, num_gpus=None):
        """
        데이터를 인터리브하여 분산 처리를 위해 분할하는 함수
        
        Args:
            block_size: 블록 크기
            num_gpus: GPU 개수
        
        Returns:
            인터리브된 데이터셋 또는 GPU별 데이터셋 리스트
        """
        keys = np.reshape(self.keys, (-1, block_size)).T
        if num_gpus is None: return self.change_keys(keys.ravel().tolist())
        ret, num_gpus = (None, num_gpus) if isinstance(num_gpus, int) else num_gpus
        keys = np.concatenate([keys, np.full((-keys.shape[0]%num_gpus, keys.shape[1]), None)])
        keys = np.reshape(keys, (keys.shape[0]//num_gpus, num_gpus, -1)).swapaxes(0, 1).reshape(num_gpus, -1)
        new_datasets = [self.change_keys(gpu_keys.tolist()) for gpu_keys in keys]
        return new_datasets if ret is None else new_datasets[ret]

    def remove(self, *datasets):
        """
        지정된 데이터셋의 키들을 제거하는 함수
        
        Args:
            *datasets: 제거할 데이터셋들
        
        Returns:
            키가 제거된 새로운 ArcDataset
        """
        remove_keys = {k for d in datasets for k in d.keys}
        new_keys = [k for k in self.keys if k not in remove_keys]
        return self.change_keys(new_keys)

    def keep_key_startswith(self, key_start):
        """
        특정 접두사로 시작하는 키만 유지하는 함수
        
        Args:
            key_start: 키 접두사
        
        Returns:
            필터링된 새로운 ArcDataset
        """
        new_keys = [k for k in self.keys if k.startswith(key_start)]
        return self.change_keys(new_keys)

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only):
        """
        단일 변환을 적용하는 함수
        
        Args:
            mod_func: 변환 함수
            descriptor: 변환 설명자
            i: 인덱스
            keep_key: 키 유지 여부
            inputs_only: 입력만 변환할지 여부
        
        Returns:
            변환된 새로운 ArcDataset
        """
        queries = {}
        replies = {}
        keys    = []
        for k0 in self.keys:
            # 변환 설명자 생성
            desc = (('copy{i}' if mod_func is np.copy else mod_func.__name__) if descriptor is None else descriptor if isinstance(descriptor, str) else descriptor(self.queries[k0])).format(i=i)
            func = lambda a, d: np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            keys.append(k1)
            # 쿼리 변환
            queries[k1] = {m: [{t: (func(a, desc) if t=='input' or not inputs_only else a) for t, a in x.items()} for x in e] for m, e in self.queries[k0].items()}
            # 답안 변환 (입력만 변환하는 경우가 아닐 때)
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
        ret = self.__class__(queries=queries, replies=replies, keys=keys)
        return ret

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False, keep_key=False, shuffle=False, join=True, inputs_only=False):
        """
        데이터셋에 변환을 적용하는 메인 함수
        
        Args:
            mod_func: 변환 함수
            descriptor: 변환 설명자
            n: 변환 횟수
            stack: 스택 여부 (None이면 자동 결정)
            keep: 원본 유지 여부
            keep_key: 키 유지 여부
            shuffle: 셞플 여부
            join: 결합 여부
            inputs_only: 입력만 변환할지 여부
        
        Returns:
            변환된 ArcDataset (또는 데이터셋 리스트)
        """
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None: stack = mod_func.__name__.startswith('rot')  # 회전의 경우 기본적으로 스택
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only)
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter):
        """
        특정 키의 데이터를 포맷된 형태로 가져오는 함수
        
        Args:
            key: 데이터 키
            formatter: 포맷터 객체
        
        Returns:
            포맷된 데이터 딕셔너리
        """
        assert formatter.out2_token is None or key in self.replies
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'], i=len(self.queries[key]['train']))
        reply = formatter.fmt_reply(self.replies[key], self.faulty.get(key)) if key in self.replies else ''
        text = train+query+reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        return dict(key=key, train=train, query=query, reply=reply, input=train+query, text=text)

    def as_list(self, formatter):
        """
        전체 데이터셋을 리스트 형태로 반환하는 함수
        
        Args:
            formatter: 포맷터 객체
        
        Returns:
            포맷된 데이터 리스트
        """
        return [self.get(key, formatter) for key in self.keys]

    def as_dataset(self):
        """HuggingFace Dataset 형태로 변환하는 함수"""
        from datasets import Dataset
        return Dataset.from_list([{'key': k, 'query': self.queries[k], 'reply': self.replies[k]} for k in self.keys])

    def get_length(self, key, formatter, name, max_of_transposed=False):
        """
        특정 키의 데이터 길이를 계산하는 함수
        
        Args:
            key: 데이터 키
            formatter: 포맷터 객체
            name: 계산할 부분 ('input' 또는 'reply')
            max_of_transposed: 전치된 버전과의 최대값 사용 여부
        
        Returns:
            데이터 길이
        """
        if formatter is None:
            # 포맷터가 없는 경우 원시 크기 계산
            if   name=='input': return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name=='reply': return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else: assert False
        else:
            # 포맷터를 사용한 토큰 길이 계산
            datasets = [self]
            if max_of_transposed:
                if self.transposed_dataset is None: self.transposed_dataset = self.mod(np.transpose, keep=False, keep_key=True)
                datasets.append(self.transposed_dataset)
            return max(len(formatter.tokenizer(ds.get(key, formatter=formatter)[name])['input_ids']) for ds in datasets)

    def get_lengths(self, formatter, name, max_of_transposed=False):
        """모든 키의 데이터 길이를 계산하는 함수"""
        return {key: self.get_length(key, formatter=formatter, name=name, max_of_transposed=max_of_transposed) for key in self.keys}

    def sorted_by_len(self, reverse=False, **kwargs):
        """길이순으로 정렬된 데이터셋을 반환하는 함수"""
        new_keys = [key for _, key in sorted([(v, k) for k, v in self.get_lengths(**kwargs).items()], reverse=reverse)]
        return self.change_keys(new_keys)

    def filter_by_len(self, min_len=0, max_len=float('inf'), **kwargs):
        """길이로 필터링된 데이터셋을 반환하는 함수"""
        new_keys = [k for k, v in self.get_lengths(**kwargs).items() if min_len<=v<=max_len]
        return self.change_keys(new_keys)

    def cut_to_query_count(self, max_count, from_end=False):
        """
        쿼리 개수를 제한하는 함수
        
        Args:
            max_count: 최대 쿼리 개수
            from_end: 끝에서부터 자를지 여부
        
        Returns:
            쿼리 개수가 제한된 새로운 ArcDataset
        """
        new_queries = {}
        for k in self.keys:
            new_queries[k] = q = self.queries[k]
            while len(q['train'])>max_count: 
                q['train'] = q['train'][:-1] if from_end else q['train'][1:]
        return self.__class__(queries=new_queries, replies=self.replies, keys=[k for k in self.keys])

    def cut_to_len(self, formatter, name, max_len, max_new_tokens='auto', from_end=False, quiet=False, **kwargs):
        """
        최대 길이에 맞춰 데이터를 자르는 함수
        
        Args:
            formatter: 포맷터 객체
            name: 계산할 부분
            max_len: 최대 길이
            max_new_tokens: 최대 새 토큰 수
            from_end: 끝에서부터 자를지 여부
            quiet: 조용한 모드 여부
        
        Returns:
            길이가 조정된 새로운 ArcDataset
        """
        if max_new_tokens:
            if max_new_tokens=='auto': max_new_tokens = formatter.max_new_tokens()
            max_len_old, max_len = max_len, max_len - max_new_tokens
            if not quiet: print(f'*** 작업 크기를 최대 {max_len_old} 토큰 ({max_len} 입력 + {max_new_tokens} 생성)으로 축소 중...')
        elif not quiet: print(f'*** 작업 크기를 최대 {max_len} 토큰으로 축소 중...')
        
        temp_ds = self.change_keys(self.keys)
        new_keys = []
        new_queries = {}
        new_replies = {}
        
        for key in (self.keys if quiet else tqdm(self.keys, file=sys.stdout)):
            reply = temp_ds.replies.get(key)
            # 길이가 초과하는 동안 예제를 제거
            while max_len<temp_ds.get_length(key, formatter=formatter, name=name, **kwargs):
                query = temp_ds.queries[key]
                if not key.split('.')[-1].startswith('ex'): 
                    key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split('.')
                assert key_split[-1].startswith('ex')
                key = '.'.join(key_split[:-1] + [f'ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}'])
                temp_ds.queries[key] = {k: ((v[:-1] if from_end else v[1:]) if k=='train' else v) for k, v in query.items()}
                if reply is not None: temp_ds.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None: new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)

    def shuffle_ex(self, perm=None, keep_max=None):
        """
        예제 순서를 섞는 함수
        
        Args:
            perm: 사용할 순열 (None이면 무작위)
            keep_max: 유지할 최대 예제 수
        
        Returns:
            예제가 섞인 새로운 ArcDataset
        """
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None: p = p[:keep_max]
            # 키에 예제 순서 정보 추가
            new_key = f'{key}.ex' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='train' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def shuffle_rp(self, keep_max=None):
        """
        테스트 케이스 순서를 섞는 함수
        
        Args:
            keep_max: 유지할 최대 테스트 케이스 수
        
        Returns:
            테스트 케이스가 섞인 새로운 ArcDataset
        """
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['test'])
            p = np.random.permutation(n)
            if keep_max is not None: p = p[:keep_max]
            # 키에 테스트 케이스 순서 정보 추가
            new_key = f'{key}.rp' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='test' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = np.array(self.replies[key], dtype=object)[p].tolist()
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def append_to_keys(self, text):
        """
        모든 키에 텍스트를 추가하는 함수
        
        Args:
            text: 추가할 텍스트
        
        Returns:
            키가 수정된 새로운 ArcDataset
        """
        return self.change_keys([f'{k}{text}' for k in self.keys])

    def random_select(self, n):
        """
        n개 그룹 중에서 무작위로 하나씩 선택하는 함수
        
        Args:
            n: 그룹 수
        
        Returns:
            무작위 선택된 새로운 ArcDataset
        """
        keys = np.array(self.keys).reshape(n, -1).T
        choice = np.random.randint(0, n, size=[len(keys)])
        return self.change_keys(keys[np.arange(len(keys)), choice])

    def augment(self, tp=False, rot=False, n=1, perm=None, perm_append=False, shfl_keys=False, shfl_ex=False, seed=None, quiet=False, inputs_only=False):
        """
        데이터 증강을 수행하는 메인 함수
        
        Args:
            tp: 전치 변환 사용 여부 ('rand'면 무작위 선택)
            rot: 회전 변환 사용 여부 ('rand'면 무작위 선택)
            n: 변환 횟수
            perm: 순열 타입 (None, 'rnd_col', 'rnd_all', 'cnt_col', 'cnt_all')
            perm_append: 순열 변환을 추가로 유지할지 여부
            shfl_keys: 키 셔플 여부
            shfl_ex: 예제 셔플 여부
            seed: 랜덤 시드
            quiet: 조용한 모드 여부
            inputs_only: 입력만 변환할지 여부
        
        Returns:
            증강된 새로운 ArcDataset
        """
        if not quiet: print(f"*** 데이터셋 증강{' (입력만)' if inputs_only else ''} 중...")
        np.random.seed(seed)
        d = self
        
        # 전치 변환
        if tp: d = d.mod(np.transpose, keep=True, inputs_only=inputs_only)
        if tp=='rand': d = d.random_select(n=2)
        
        # 회전 변환
        if rot: d = d.mod(np.rot90, n=3, keep=True, inputs_only=inputs_only)
        if rot=='rand': d = d.random_select(n=4)
        
        # 순열 변환
        if perm is None and n<=1: d = d.shuffled() if shfl_keys else d
        else: d = d.mod(*([np.copy] if perm is None else globals()[f"permute_{perm}"]), n=n, shuffle=shfl_keys, keep=perm_append, inputs_only=inputs_only)
        
        # 예제 셔플
        np.random.seed(seed)
        if shfl_ex: d = d.shuffle_ex()
        return d

    def remove_replies(self):
        """답안을 제거한 새로운 데이터셋을 반환하는 함수"""
        return self.__class__(queries=self.queries, replies={}, keys=[k for k in self.keys])

    def split_at_pos(self, pos, random_seed=None):
        """
        지정된 위치에서 데이터셋을 분할하는 함수
        
        Args:
            pos: 분할 위치 (정수 또는 비율)
            random_seed: 랜덤 시드 (섞기용)
        
        Returns:
            분할된 두 개의 ArcDataset 튜플
        """
        keys = self.keys
        if random_seed is not None:
            np.random.seed(random_seed)
            keys = np.random.permutation(keys)
        if isinstance(pos, float): pos = int(pos * len(self.keys) + 0.5)
        keys_split = [keys[:pos], keys[pos:]]
        return tuple(self.change_keys(new_keys, keep_flags=True) for new_keys in keys_split)

    def get_submission(self, results=None):
        """
        제출용 형식의 결과를 생성하는 함수
        
        Args:
            results: 결과 딕셔너리 (선택사항)
        
        Returns:
            제출용 형식의 딕셔너리
        """
        assert self.is_orig==True, '원본 데이터셋에서만 실행해야 합니다.'
        # 각 문제마다 2번의 시도 기회를 가진 제출 형식 생성
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(self.queries[k]['test']))] for k in self.keys}
        if results is not None: self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        """
        결과를 제출 형식에 채우는 정적 메서드
        
        Args:
            results: 결과 딕셔너리
            submission: 제출 형식 딕셔너리
        """
        print(f'*** {len(results)}개 출력에 대한 제출 생성 중...')
        for k, v in results.items():
            base_id, base_nr = k.split('_')
            target_dict = submission[base_id][int(base_nr)]
            for i, g in enumerate(v[:len(target_dict)]):
                target_dict[f'attempt_{i+1}'] = g.tolist()

    def validate_submission(self, submission):
        """
        제출 결과를 검증하는 함수
        
        Args:
            submission: 제출 딕셔너리
        
        Returns:
            점수 (0~1 사이의 값)
        """
        assert self.is_orig==True, '원본 데이터셋에서만 실행해야 합니다.'
        score = 0
        for k, v in self.replies.items():
            for i, r in enumerate(v):
                # 두 번의 시도 중 하나라도 맞으면 점수 획득
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score

def get_class_MyDataCollator(cache=[]):
    """
    커스텀 데이터 콜레이터 클래스를 반환하는 함수 (싱글톤 패턴)
    
    Args:
        cache: 캐시 리스트 (싱글톤 구현용)
    
    Returns:
        MyDataCollator 클래스
    """
    if not cache:
        from trl import DataCollatorForCompletionOnlyLM
        
        class MyDataCollator(DataCollatorForCompletionOnlyLM):
            """ARC 작업에 특화된 커스텀 데이터 콜레이터"""
            
            def setup(self, out2_token_id=None, fault_token_id=None, fault_freq=0, sample_tries=8, mask_first_output=False):
                """
                데이터 콜레이터 설정
                
                Args:
                    out2_token_id: 두 번째 출력 토큰 ID
                    fault_token_id: 오류 토큰 ID
                    fault_freq: 오류 주입 빈도
                    sample_tries: 샘플링 시도 횟수
                    mask_first_output: 첫 번째 출력 마스킹 여부
                
                Returns:
                    설정된 self 객체
                """
                self.out2_token_id = out2_token_id
                self.fault_token_id = fault_token_id
                self.fault_freq = fault_freq
                self.sample_tries = sample_tries
                self.mask_first_output = mask_first_output
                return self

            def torch_call(self, examples):
                """
                배치 처리 메인 함수
                
                Args:
                    examples: 예제 리스트
                
                Returns:
                    처리된 배치
                """
                batch = super().torch_call(examples)
                
                # 두 번째 출력 토큰 처리
                if self.out2_token_id is not None:
                    assert not self.fault_freq
                    for i in range(len(batch['input_ids'])):
                        end_pos = ((batch['labels'][i] != -100              ).nonzero().max()).item() + 1
                        mid_pos = ((batch['labels'][i] == self.out2_token_id).nonzero().max()).item() + 1
                        beg_pos = mid_pos - (end_pos - mid_pos)
                        # 첫 번째 출력을 두 번째 출력으로 복사
                        batch['labels'][i][beg_pos:mid_pos] = batch['labels'][i][mid_pos:end_pos]
                
                # 오류 주입 처리
                elif self.fault_freq:
                    for i in range(len(batch['input_ids'])):
                        end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
                        
                        # 동적 오류 빈도 계산
                        if not isinstance(self.fault_freq, float):
                            eos_token_id = batch['labels'][i][end_pos - 1]
                            num_examples = (batch['labels'][i] == eos_token_id).sum().item() - 1
                            fault_freq = self.fault_freq[num_examples]
                        else: 
                            fault_freq = self.fault_freq
                        
                        # 확률적 오류 주입
                        if random.random() < fault_freq:
                            beg_pos = ((batch['labels'][i][:end_pos]==-100).nonzero().max()).item() + 1
                            fault_pos = random.randint(beg_pos, end_pos-2)
                            fault_tok = batch['labels'][i][fault_pos].item()
                            
                            # 다른 토큰으로 교체 시도
                            for t in range(self.sample_tries):
                                new_tok = batch['labels'][i][random.randint(beg_pos, end_pos-2)].item()
                                if fault_tok!=new_tok:
                                    batch['input_ids'][i][fault_pos] = new_tok
                                    # 오류 후 모든 토큰을 오류 토큰으로 마스킹
                                    batch['labels'][i][fault_pos+1:end_pos] = self.fault_token_id
                                    break
                
                # 첫 번째 출력 마스킹
                for i in range(len(batch['labels'])):
                    for _ in range(self.mask_first_output):
                        beg_pos = ((batch['labels'][i] != -100).nonzero().min()).item()
                        mid_pos = ((batch['labels'][i][beg_pos:] == -100).nonzero().min()).item() + beg_pos
                        end_pos = ((batch['labels'][i] != -100).nonzero().max()).item() + 1
                        if mid_pos<end_pos: batch['labels'][i][beg_pos:mid_pos] = -100
                return batch
        cache.append(MyDataCollator)
    return cache[0]

class ArcFormatter(object):
    """ARC 데이터를 텍스트 형식으로 포맷팅하는 클래스"""
    
    def __init__(self, inp_prefix, out_prefix, arr_sep, out2_use=False, out2_token=None, arr_beg='', arr_end='', pretext='', pre_out=None, exa_sep='', exa_end='', qry_prefix=None, rpl_prefix=None, rpl_sep=None, dec_sep=None, min_wid=0, min_pad='', pretext_corpus_split='', masking=0, tokenizer=None, collator_kwargs={}, repeat_input_aug=None, repeat_input_pre=None):
        """
        ArcFormatter 초기화
        
        Args:
            inp_prefix: 입력 접두사
            out_prefix: 출력 접두사
            arr_sep: 배열 분리자
            out2_use: 두 번째 출력 사용 여부
            out2_token: 두 번째 출력 토큰
            arr_beg: 배열 시작 문자
            arr_end: 배열 끝 문자
            pretext: 전문(前文)
            pre_out: 출력 전 텍스트
            exa_sep: 예제 분리자
            exa_end: 예제 끝 문자
            qry_prefix: 쿼리 접두사
            rpl_prefix: 응답 접두사
            rpl_sep: 응답 분리자
            dec_sep: 디코딩 분리자
            min_wid: 최소 너비
            min_pad: 최소 패딩 문자
            pretext_corpus_split: 전문 코퍼스 분할 문자
            masking: 마스킹 모드
            tokenizer: 토크나이저
            collator_kwargs: 콜레이터 인자들
            repeat_input_aug: 입력 반복 증강 함수
            repeat_input_pre: 입력 반복 접두사
        """
        self.tokenizer = tokenizer
        self.inp_prefix = inp_prefix
        self.out_prefix = out_prefix
        self.out2_token = out2_token
        self.out2_use = out2_use
        assert not out2_use or out2_token is not None
        assert not out2_use or masking in [1, 2]
        assert masking!=2 or out2_use or rpl_prefix is not None
        
        # 기본값 설정
        self.qry_prefix = qry_prefix if qry_prefix is not None else inp_prefix
        self.rpl_prefix = rpl_prefix if rpl_prefix is not None else out_prefix
        self.rpl_sep = rpl_sep if rpl_sep is not None else self.rpl_prefix
        self.arr_sep = arr_sep
        self.arr_beg = arr_beg
        self.arr_end = arr_end
        self.pretext = pretext
        self.pre_out = pre_out
        self.pre_out_empty = ['']*99  # 빈 출력 전 텍스트
        self.pretext_corpus_split = pretext_corpus_split
        self.exa_sep = exa_sep
        self.exa_end = exa_end
        self.dec_sep = arr_sep if dec_sep is None else dec_sep
        self.min_wid = min_wid
        self.min_pad = min_pad
        self.masking = masking
        self.collator_kwargs = collator_kwargs
        self.repeat_input_aug = repeat_input_aug
        self.repeat_input_pre = repeat_input_pre

    def fmt_array(self, array):
        """
        2D 배열을 텍스트 형식으로 포맷팅하는 함수
        
        Args:
            array: 2D 배열
        
        Returns:
            포맷된 텍스트 문자열
        """
        return self.arr_beg + self.arr_sep.join(
            str(row).replace(' ', '').replace(',', '').replace('[', '').replace(']', '') + 
            self.min_pad*max(0, self.min_wid-len(row)) 
            for row in array
        ) + self.arr_end

    def get_pre_out(self, pretext_split):
        """
        출력 전 텍스트를 가져오는 함수
        
        Args:
            pretext_split: 전문 분할 여부
        
        Returns:
            출력 전 텍스트 리스트
        """
        if self.pre_out is None: return self.pre_out_empty
        if pretext_split: return [self.pretext_corpus_split.join(list(p) + ['']) for p in self.pre_out]
        return self.pre_out

    def fmt_train(self, train, last_is_challenge=False, pretext_split=False):
        """
        훈련 예제들을 포맷팅하는 함수
        
        Args:
            train: 훈련 예제 리스트
            last_is_challenge: 마지막이 도전 문제인지 여부
            pretext_split: 전문 분할 여부
        
        Returns:
            포맷된 훈련 데이터 문자열
        """
        po = self.get_pre_out(pretext_split=pretext_split)
        ex = []
        for i, x in enumerate(train):
            if last_is_challenge and i+1==len(train):
                # 마지막이 도전 문제인 경우
                formatted_ex = f"{self.fmt_query([x], i, pretext_split=pretext_split)}{self.fmt_reply([x['output']])}"
            else:
                # 일반 훈련 예제
                formatted_ex = f"{self.inp_prefix}{self.fmt_array(x['input'])}{self.repeat_input(x, no_aug=pretext_split)}{po[i]}{self.out_prefix}{self.fmt_array(x['output'])}"
            ex.append(formatted_ex)
        
        pre = self.pretext_corpus_split.join(list(self.pretext)+['']) if pretext_split else self.pretext
        end = '' if last_is_challenge else (self.exa_end + self.tokenizer.eos_token)
        return pre + (self.exa_end + self.tokenizer.eos_token + self.exa_sep).join(ex) + end

    def fmt_query(self, query, i, pretext_split=False):
        """
        쿼리를 포맷팅하는 함수
        
        Args:
            query: 쿼리 리스트
            i: 인덱스
            pretext_split: 전문 분할 여부
        
        Returns:
            포맷된 쿼리 문자열
        """
        po = self.get_pre_out(pretext_split=pretext_split)
        return ''.join(f"{self.qry_prefix}{self.fmt_array(x['input'])}{self.repeat_input(x, no_aug=pretext_split)}{po[i]}{self.rpl_prefix}" for x in query[:1])

    def repeat_input(self, x, no_aug=False):
        """
        입력 반복 기능
        
        Args:
            x: 입력 데이터
            no_aug: 증강 없음 여부
        
        Returns:
            반복된 입력 문자열
        """
        if self.repeat_input_aug is None: return ''
        return f"{self.repeat_input_pre}{self.fmt_array(((lambda x: x) if no_aug else self.repeat_input_aug)(x['input']))}"

    def fmt_reply(self, reply, fault=None):
        """
        응답을 포맷팅하는 함수
        
        Args:
            reply: 응답 리스트
            fault: 오류 데이터 (선택사항)
        
        Returns:
            포맷된 응답 문자열
        """
        ids = self.fmt_array(reply[0]) + self.exa_end + self.tokenizer.eos_token
        if self.out2_use:
            # 두 번째 출력 사용 시
            if fault is None: fault = reply
            ids = self.fmt_array(fault[0]) + self.exa_end + self.out2_token + ids
        return ids

    def quick_test(self, decoded, done):
        """
        디코딩된 결과에 대한 빠른 테스트
        
        Args:
            decoded: 디코딩된 문자열
            done: 완료 여부
        
        Returns:
            테스트 통과 여부
        """
        sp = decoded.split(self.tokenizer.eos_token)[0].split(self.dec_sep)
        sl = len(sp[0])
        is_prefix = sl>0 and len(sp[-1])<=sl and (len(sp)==1 or len(sp[-2])==sl) and all(x.isdigit() for x in sp[-1])
        return is_prefix and (not done or len(sp[-1])==0 or len(sp[-1])==sl)

    @staticmethod
    def is_valid_solution(guess):
        """
        추측이 유효한 솔루션인지 확인하는 정적 메서드
        
        Args:
            guess: 추측 배열
        
        Returns:
            유효성 여부
        """
        return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)

    def max_new_tokens(self, safety_margin=1):
        """
        최대 새 토큰 수를 계산하는 함수
        
        Args:
            safety_margin: 안전 마진
        
        Returns:
            최대 새 토큰 수
        """
        # 최대 크기 응답 (30x30)으로 계산
        max_sized_reply = np.zeros([30, 30], dtype=int)
        tokenized = self.tokenizer(self.fmt_reply([max_sized_reply]))['input_ids']
        max_new_tokens = len(tokenized)
        if tokenized[0]==self.tokenizer.bos_token_id: max_new_tokens -= 1
        return max_new_tokens + safety_margin

    def de_tokenize(self, tokens, scores=None):
        """
        토큰을 디토크나이즈하는 함수
        
        Args:
            tokens: 토큰 배열
            scores: 점수 배열 (선택사항)
        
        Returns:
            (출력 길이, 점수 값, 디토크나이즈된 텍스트, 점수들) 튜플
        """
        import torch
        tokens_cut = cut_at_token(tokens, self.tokenizer.eos_token_id)
        de_tokenized = self.tokenizer.batch_decode([tokens_cut])[0]
        score_val = None
        
        if scores is not None:
            tokens_with_eos = tokens[:len(tokens_cut)+1]
            # 로그 소프트맥스로 점수 값 계산
            score_val = torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1).numpy().copy()[np.arange(len(tokens_with_eos)), tokens_with_eos].sum()
            
            # 숫자 토큰들만 추출
            number_token_ids = [self.tokenizer.vocab[k] for k in map(str, range(10))]
            fault_token_id = self.collator_kwargs.get('fault_token_id')
            if fault_token_id is not None: number_token_ids.append(fault_token_id)
            number_token_ids = np.array(number_token_ids)
            number_positions = (tokens_cut[..., np.newaxis] == number_token_ids).any(-1)
            scores = scores[:len(tokens_cut), number_token_ids][number_positions]
            scores = torch.nn.functional.log_softmax(torch.tensor(scores), dim=-1)[:, :10].numpy().copy()
        
        return max(len(tokens)+1, len(tokens_cut)), score_val, de_tokenized, scores

    def decode_to_array_single(self, text, score=None, limit_rows=30):
        """
        단일 텍스트를 배열로 디코딩하는 함수
        
        Args:
            text: 디코딩할 텍스트
            score: 점수 배열 (선택사항)
            limit_rows: 최대 행 수 제한
        
        Returns:
            디코딩 결과 딕셔너리
        """
        try:
            # 텍스트를 행별로 분할하고 숫자만 추출
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in text.split(self.dec_sep)] if len(row)]
            if limit_rows and len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
                limited = True
            else: 
                limited = False
            
            decoded = np.array(by_rows, dtype=int)
            if self.is_valid_solution(decoded):
                try:
                    assert score is not None
                    decoded_flat = decoded.ravel()
                    if limited: score = score[:len(decoded_flat)]
                    
                    # 다양한 점수 형태 계산
                    score_all = score.reshape(decoded.shape + score.shape[1:])
                    score_result = score[range(len(decoded_flat)), decoded_flat]
                    score_reshaped = score_result.reshape(decoded.shape)
                    score_cum_reshaped = score_result.cumsum().reshape(score_reshaped.shape)
                    score_all_cum = score_cum_reshaped[..., np.newaxis] - score_reshaped[..., np.newaxis] + score_all
                except: 
                    # 점수 계산 실패 시 무한대 값으로 채움
                    score_reshaped = score_cum_reshaped = np.full(decoded.shape, -float('inf'))
                
                return {
                    'output': decoded, 
                    'score': score_reshaped, 
                    'score_cum': score_cum_reshaped, 
                    'score_all': score_all, 
                    'score_all_cum': score_all_cum
                }
        except: 
            pass
        return {}

    def decode_to_array(self, text, score=None, limit_rows=30):
        """
        텍스트를 배열로 디코딩하는 메인 함수
        
        Args:
            text: 디코딩할 텍스트
            score: 점수 배열 (선택사항)
            limit_rows: 최대 행 수 제한
        
        Returns:
            디코딩 결과 리스트
        """
        if not self.out2_use: 
            text, score = [text], [score]
        else:
            # 두 번째 출력 토큰으로 분할
            text = text.split(self.out2_token)
            if score is None: 
                score = [None]*len(text)
            else:
                # 텍스트 길이에 따라 점수 분할
                lengths = np.cumsum([len(list(filter(str.isdigit, t))) for t in text])
                score = [score[s:e] for s, e in zip([0]+lengths[:-1].tolist(), lengths)]
        
        return [self.decode_to_array_single(t, s) for t, s in zip(text, score)]

    def get_corpus(self):
        """
        토크나이저 학습용 코퍼스를 생성하는 함수
        
        Returns:
            코퍼스 텍스트
        """
        try:
            old_min_wid, self.min_wid = self.min_wid, min(self.min_wid, 2)
            # 0-9 숫자로 구성된 간단한 예제 생성
            return self.fmt_train([{'input': [[i] for i in range(10)], 'output': [[i] for i in range(10)]}]*3, last_is_challenge=True, pretext_split=True)
        finally: 
            self.min_wid = old_min_wid

    def get_data_collator(self):
        """
        데이터 콜레이터를 생성하는 함수
        
        Returns:
            데이터 콜레이터 객체 또는 None
        """
        if not self.masking: return None
        
        from transformers import DataCollatorForLanguageModeling
        collator_params = dict(tokenizer=self.tokenizer, mlm=False)
        
        # 두 번째 출력 토큰 ID 설정
        pass_out2_token = self.tokenizer.vocab[self.out2_token] if self.out2_use and self.masking==1 else None
        
        if self.masking:
            assert not self.collator_kwargs.get('mask_first_output') or self.masking==1
            # 커스텀 콜레이터 생성
            data_collator = get_class_MyDataCollator()(
                **collator_params,
                instruction_template=[self.inp_prefix, self.tokenizer.bos_token][self.masking - 1],
                response_template=[self.out_prefix, (self.out2_token if self.out2_use else self.rpl_sep)][self.masking - 1],
            ).setup(out2_token_id=pass_out2_token, **self.collator_kwargs)
        else:
            assert not self.collator_kwargs, '마스킹이 켜져있을 때만 지원됩니다'
            data_collator = DataCollatorForLanguageModeling(**collator_params)
        
        return data_collator

    def get_output_token_ids(self):
        """
        출력에 사용되는 토큰 ID들을 반환하는 함수
        
        Returns:
            출력 토큰 ID 리스트
        """
        assert not self.out2_use
        # 숫자 토큰들 (0-9)
        num_tokens = [self.tokenizer.vocab[str(i)] for i in range(10)]
        
        # 분리자 토큰들
        sep_tokens = []
        for txt in [self.arr_beg, self.arr_sep, self.arr_end, self.exa_sep]:
            if txt:
                for tok in self.tokenizer(txt)['input_ids'][1:]:
                    sep_tokens.append(tok)
        sep_tokens.append(self.tokenizer.eos_token_id)
        
        return num_tokens + sorted(set(sep_tokens))

# 사전 정의된 포맷터들
ArcFormatter_pretext2 = lambda **kwargs: ArcFormatter(
    masking=1, 
    inp_prefix='I', 
    out_prefix='O', 
    arr_sep='\n', 
    arr_end='\n', 
    pretext='ABCDEFGHJKLMNPQRSTUVWXYZ',  # I와 O를 제외한 알파벳
    pretext_corpus_split='\n', 
    **kwargs
)

ArcFormatter_pretext3 = lambda **kwargs: ArcFormatter(
    masking=1, 
    inp_prefix='I', 
    out_prefix='O', 
    arr_sep='\n', 
    arr_end='\n', 
    pretext='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',  # 대소문자 알파벳 (I, O, i, o 제외)
    pretext_corpus_split='\n', 
    **kwargs
)

ArcFormatter_premix_2 = lambda **kwargs: ArcFormatter(
    masking=1, 
    inp_prefix='I', 
    out_prefix='O', 
    arr_sep='\n', 
    arr_end='\n', 
    pretext='ABCDEFGHJKLMNPQRSTUVWXYZ', 
    pre_out=['+/-=']*99,  # 출력 전에 수학 기호 추가
    pretext_corpus_split='\n', 
    **kwargs
)

ArcFormatter_premix_3 = lambda **kwargs: ArcFormatter(
    masking=1, 
    inp_prefix='I', 
    out_prefix='O', 
    arr_sep='\n', 
    arr_end='\n', 
    pretext='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz', 
    pre_out=['+/-=']*99,  # 출력 전에 수학 기호 추가
    pretext_corpus_split='\n', 
    **kwargs
)

# 사용 가능한 포맷터들의 딕셔너리
available_formatters = dict(
    ArcFormatter_pretext2=ArcFormatter_pretext2,
    ArcFormatter_pretext3=ArcFormatter_pretext3,
    ArcFormatter_premix_2=ArcFormatter_premix_2,
    ArcFormatter_premix_3=ArcFormatter_premix_3,
)