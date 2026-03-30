sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    libjpeg-dev libpng-dev libtiff-dev
apt install libimage-exiftool-perl -y


mkdir NeuroGraphia && cd NeuroGraphia
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Visão computacional e imagem
pip install opencv-python-headless numpy Pillow

# PyTorch — versão CPU (menor download, ~200 MB vs ~2 GB da versão GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Kraken (binarização + segmentação)
pip install "kraken>=4.3"

# TrOCR
pip install transformers



# Para processar um único manuscrito com o modelo de alta precisão
python3 neurog.py sample.jpeg --model large --device gpu

# Para processar uma pasta inteira otimizando para CPU
python3 neurog.py --folder ./meus_manuscritos --quantize --batch 4


python3 neurog.py sample.jpeg --quantize --batch 4