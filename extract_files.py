#!/usr/bin/env python3
"""
원본 노트북에서 Python 파일들을 추출하는 도우미 스크립트
"""

import json
import sys
from pathlib import Path

def extract_writefile_cells(notebook_path):
    """Jupyter 노트북에서 %%writefile 셀들을 추출"""
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    files_to_write = {}
    
    for cell in notebook.get('cells', []):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            
            # %%writefile로 시작하는 셀 찾기
            if source.startswith('%%writefile'):
                lines = source.split('\n')
                first_line = lines[0]
                
                # 파일명 추출
                if ' ' in first_line:
                    filename = first_line.split(' ', 1)[1].strip()
                    # 파일 내용 (첫 줄 제외)
                    content = '\n'.join(lines[1:])
                    files_to_write[filename] = content
    
    return files_to_write

def main():
    if len(sys.argv) < 2:
        print("사용법: python extract_files.py <notebook.ipynb>")
        sys.exit(1)
    
    notebook_path = Path(sys.argv[1])
    if not notebook_path.exists():
        print(f"파일을 찾을 수 없습니다: {notebook_path}")
        sys.exit(1)
    
    print(f"노트북에서 파일 추출 중: {notebook_path}")
    
    try:
        files = extract_writefile_cells(notebook_path)
        
        for filename, content in files.items():
            output_path = Path(filename)
            
            # 디렉토리 생성
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 파일 쓰기
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"생성됨: {output_path}")
        
        print(f"\n총 {len(files)}개 파일이 추출되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
