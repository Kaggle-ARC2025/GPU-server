#!/bin/bash
# ARC Prize 2025 로컬 실행을 위한 셋업 스크립트

echo "ARC Prize 2025 로컬 환경 설정"
echo "============================"

# 현재 디렉토리 확인
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 디렉토리 구조 생성
echo "1. 디렉토리 구조 생성 중..."
mkdir -p input/arc-prize-2025
mkdir -p input/wb55l_nemomini_fulleval/transformers/default/1
mkdir -p input/unsloth-2024-9-post4/wheelhouse
mkdir -p temp/checkpoints

# Python 환경 확인
echo -e "\n2. Python 환경 확인..."
python --version

# 가상환경 생성 제안
echo -e "\n3. 가상환경 생성 (권장)"
echo "python -m venv venv"
echo "source venv/bin/activate  # Linux/Mac"
echo "# 또는"
echo "venv\\Scripts\\activate  # Windows"

# 필수 패키지 설치 명령
echo -e "\n4. 필수 패키지 설치 명령:"
cat > requirements.txt << 'EOF'
torch>=2.0.0
transformers>=4.30.0
datasets>=2.10.0
tqdm>=4.65.0
numpy>=1.24.0
peft>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.40.0
tokenizers>=0.13.0
huggingface-hub>=0.16.0
EOF

echo "pip install -r requirements.txt"

# 데이터 다운로드 안내
echo -e "\n5. 필요한 데이터 파일:"
echo "   a) ARC 데이터셋 (Kaggle에서 다운로드):"
echo "      - arc-agi_test_challenges.json"
echo "      - arc-agi_training_solutions.json"
echo "      위 파일들을 input/arc-prize-2025/ 에 복사"
echo ""
echo "   b) 사전 훈련된 모델 (wb55l_nemomini_fulleval):"
echo "      HuggingFace 또는 Kaggle에서 다운로드하여"
echo "      input/wb55l_nemomini_fulleval/transformers/default/1/ 에 복사"
echo ""
echo "   c) Unsloth (선택사항):"
echo "      wheel 파일이 있다면 input/unsloth-2024-9-post4/wheelhouse/ 에 복사"
echo "      없으면 pip로 직접 설치됩니다"

# 실행 방법 안내
echo -e "\n6. 실행 방법:"
echo "   python main_local.py"

echo -e "\n설정이 완료되었습니다!"
