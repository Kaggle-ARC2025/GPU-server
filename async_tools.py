# async_tools.py
# 전체 내용은 원본 노트북의 %%writefile async_tools.py 셀에서 복사해주세요.
# 이 파일은 비동기 처리 관련 도구들을 포함합니다.
import sys
import asyncio

async def stream_reader(stream, id, to):
    """
    스트림에서 데이터를 비동기적으로 읽고 실시간으로 출력하는 함수
    
    Args:
        stream: 읽을 스트림 (stdout 또는 stderr)
        id: 프로세스 식별자 (None이면 ID 표시 안함)
        to: 출력할 대상 스트림 (None이면 출력 안함)
    """
    # ID 접두사 설정 (None이면 빈 문자열)
    id = '' if id is None else f'{id}. '
    data = b''  # 아직 완성되지 않은 라인의 버퍼
    
    while True:
        # 스트림에서 최대 4096바이트씩 읽기
        read = await stream.read(n=4096)
        if not read: break  # 더 이상 읽을 데이터가 없으면 종료
        
        if to is not None:
            # 개행 문자로 라인 분할 (마지막에 'X' 추가하여 빈 라인 구분)
            *complete_lines, data = (data + read + b'X').splitlines()
            data = data[:-1]  # 'X' 제거
            
            # 완성된 라인들을 출력
            for line in complete_lines:
                line = line.rstrip()  # 끝의 공백 문자 제거
                if line:  # 빈 라인이 아닌 경우에만 출력
                    print(f"{id}{line.decode('utf-8')}", file=to, end='\n', flush=True)

async def wait_for_subprocess(subprocess, print_output=False, id=None):
    """
    단일 서브프로세스의 완료를 기다리면서 출력을 스트리밍하는 함수
    
    Args:
        subprocess: 기다릴 서브프로세스 객체
        print_output: 출력을 콘솔에 표시할지 여부
        id: 프로세스 식별자 (다중 프로세스 실행 시 구분용)
    
    Returns:
        서브프로세스의 종료 코드
    """
    # stdout과 stderr를 동시에 비동기적으로 처리
    await asyncio.gather(
        stream_reader(
            subprocess.stdout, 
            id, 
            (sys.stdout if print_output else None)  # 출력 표시 여부에 따라 대상 설정
        ),
        stream_reader(
            subprocess.stderr, 
            id, 
            (sys.stderr if print_output else None)   # 에러 출력도 동일하게 처리
        ),
    )
    
    # 서브프로세스의 종료를 기다리고 종료 코드 반환
    return await subprocess.wait()

async def wait_for_subprocesses(*processes, print_output=False):
    """
    여러 서브프로세스들의 완료를 동시에 기다리는 함수
    
    Args:
        *processes: 기다릴 서브프로세스들 (가변 인수)
        print_output: 출력을 콘솔에 표시할지 여부
    
    Returns:
        모든 서브프로세스들의 종료 코드 리스트
    """
    # 각 프로세스에 대해 wait_for_subprocess를 비동기적으로 실행
    # 프로세스가 여러 개인 경우에만 ID 부여 (구분용)
    return await asyncio.gather(*[
        wait_for_subprocess(
            p, 
            print_output=print_output, 
            id=i if len(processes) > 1 else None  # 단일 프로세스면 ID 없음
        ) 
        for i, p in enumerate(processes)
    ])