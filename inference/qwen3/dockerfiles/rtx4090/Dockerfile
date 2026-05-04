FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST=8.9

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libjpeg-dev \
    zlib1g-dev \
    git \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

WORKDIR /app

RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

COPY dockerfiles/rtx4090/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN ln -s /usr/bin/python3.10 /usr/bin/python

ENV WANDB_API_KEY=<your_token>

RUN chmod +x src/start.sh
CMD ["bash", "src/start.sh"]