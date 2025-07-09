# selection.py
# 전체 내용은 원본 노트북의 %%writefile selection.py 셀에서 복사해주세요.
# 이 파일은 결과 선택 알고리즘들을 포함합니다.
import numpy as np

def hashable(guess):
    """
    2D 배열을 해시 가능한 튜플로 변환하는 함수
    
    Args:
        guess: 2D numpy 배열 형태의 추측
    
    Returns:
        중첩 튜플 형태의 해시 가능한 객체
    """
    return tuple(map(tuple, guess))

def make_unique(guess_list, indices=None):
    """
    추측 리스트에서 중복을 제거하는 함수
    
    Args:
        guess_list: 추측들의 리스트
        indices: 인덱스 리스트 (선택사항)
    
    Returns:
        중복이 제거된 추측 리스트 (또는 추측과 인덱스 튜플)
    """
    used = set()  # 이미 사용된 해시값들을 저장
    out = []      # 고유한 추측들
    out_ind = []  # 고유한 추측들의 인덱스
    
    for i, g in enumerate(guess_list):
        h = hashable(g)
        if h not in used:
            used.add(h)
            out.append(np.array(g))
            if indices is not None: 
                out_ind.append(indices[i])
    
    return out if indices is None else (out, out_ind)

def first_only(guesses):
    """
    첫 번째 추측만 반환하는 선택 알고리즘
    
    Args:
        guesses: 추측 딕셔너리 {key: {'output': array, ...}}
    
    Returns:
        첫 번째 추측만 포함된 리스트
    """
    return [g['output'] for g in guesses.values()][:1]

def keep_order(guesses):
    """
    모든 추측을 원래 순서대로 유지하는 선택 알고리즘
    
    Args:
        guesses: 추측 딕셔너리
    
    Returns:
        모든 추측의 출력 배열 리스트
    """
    return [g['output'] for g in guesses.values()]

def keep_order_unique(guesses):
    """
    원래 순서를 유지하면서 중복을 제거하는 선택 알고리즘
    
    Args:
        guesses: 추측 딕셔너리
    
    Returns:
        중복이 제거된 고유한 추측들의 리스트
    """
    return make_unique(keep_order(guesses))

def get_best_shape_by_score(guess_list, getter, once_per_result=True):
    """
    점수 기반으로 최고의 형태(shape)를 찾는 함수
    
    Args:
        guess_list: 추측 리스트
        getter: 점수를 계산하는 함수
        once_per_result: 동일한 결과당 한 번만 계산할지 여부
    
    Returns:
        (최고 점수, 형태, 인덱스들) 튜플
    """
    seen_outputs = set()  # 이미 본 출력들
    shape_scores = {}     # 형태별 점수와 인덱스 저장
    
    for i, g in enumerate(guess_list):
        shape = tuple(g['output'].shape)  # 배열의 형태 (높이, 너비)
        scores = shape_scores[shape] = shape_scores.get(shape, [[], []])
        scores[1].append(i)  # 인덱스 추가
        
        h = hashable(g['output'])
        if h in seen_outputs: continue
        if once_per_result: seen_outputs.add(h)
        scores[0].append(g)  # 추측 추가
    
    # 각 형태별로 점수 계산 후 정렬
    shape_scores = [(getter(scores), shape, indices) for shape, (scores, indices) in shape_scores.items()]
    shape_scores = sorted(shape_scores, key=(lambda x: x[0]), reverse=True)
    return shape_scores[0]  # 최고 점수의 형태 반환

def score_sum(guesses, getter, shape_getter=None, prefer_common_shape=True):
    """
    점수 합계를 기반으로 추측을 정렬하는 일반적인 함수
    
    Args:
        guesses: 추측 딕셔너리
        getter: 점수를 계산하는 함수
        shape_getter: 형태 점수를 계산하는 함수 (기본값: getter와 동일)
        prefer_common_shape: 일반적인 형태를 선호할지 여부
    
    Returns:
        점수순으로 정렬된 출력 배열들의 리스트
    """
    if shape_getter is None: shape_getter = getter
    guess_list = list(guesses.values())
    
    # 일반적인 형태를 선호하는 경우, 해당 인덱스들을 찾음
    common_shape_indices = set(get_best_shape_by_score(guess_list, shape_getter)[2]) if prefer_common_shape else []
    
    scores = {}
    for i, g in enumerate(guess_list):
        h = hashable(g['output'])
        # [일반적인_형태_여부, 추측들, 출력_배열]
        x = scores[h] = scores.get(h, [i in common_shape_indices, [], g['output']])
        x[1].append(g)
    
    # 점수 계산 및 정렬: (일반적인_형태_여부, 계산된_점수, 출력_배열)
    scores = [(cs, getter(sc), o) for cs, sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[:2]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
    return ordered_outputs

# 확률 합계를 계산하는 getter 함수
getter_all_probsum = lambda guesses: sum(np.exp(g['score_val']) for g in guesses)

def score_all_probsum(guesses): 
    """
    모든 확률의 합계를 기반으로 추측을 선택하는 알고리즘
    
    Args:
        guesses: 추측 딕셔너리
    
    Returns:
        확률 합계순으로 정렬된 추측들
    """
    return score_sum(guesses, getter_all_probsum)

def getter_full_probmul(p):
    """
    전체 확률 곱셈을 위한 getter 생성 함수
    
    Args:
        p: 기준선(baseline) 값
    
    Returns:
        확률 곱셈을 계산하는 getter 함수
    """
    def _getter(guesses, baseline=p):
        """
        추론 점수와 증강 점수를 결합하여 전체 점수를 계산
        
        Args:
            guesses: 추측 리스트
            baseline: 기준선 값
        
        Returns:
            결합된 점수
        """
        # 추론 점수: 각 추측의 점수에 기준선을 더한 합
        inf_score = sum([g['score_val']+baseline for g in guesses])
        
        # 증강 점수: 다중 점수들의 평균 (기준선 포함)
        aug_score = np.mean([sum(s+baseline for s in g['score_multi_nl']) for g in guesses])
        
        return inf_score + aug_score
    return _getter

def score_full_probmul_3(guesses): 
    """
    기준선 3을 사용한 전체 확률 곱셈 기반 선택 알고리즘
    
    Args:
        guesses: 추측 딕셔너리
    
    Returns:
        전체 확률 곱셈 점수순으로 정렬된 추측들
    """
    return score_sum(guesses, getter_full_probmul(3), prefer_common_shape=False)

# 사용 가능한 선택 알고리즘들의 리스트
selection_algorithms = [
    first_only,            # 첫 번째만 선택
    keep_order,            # 순서 유지
    keep_order_unique,     # 순서 유지 + 중복 제거
    score_all_probsum,     # 확률 합계 기반
    score_full_probmul_3,  # 전체 확률 곱셈 기반 (기준선=3)
]