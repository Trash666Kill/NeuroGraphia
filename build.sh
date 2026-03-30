sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    libjpeg-dev libpng-dev libtiff-dev

mkdir NeuroGraphia && cd NeuroGraphia
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Visão computacional e imagem
pip install opencv-python-headless numpy Pillow

# Kraken (binarização + segmentação)
pip install kraken

# PyTorch — versão CPU (menor download, ~200 MB vs ~2 GB da versão GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# TrOCR
pip install transformers
