apt install python3-pip
pip install uv --break-system-packages
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install torch numpy
