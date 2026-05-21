FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV TORCH_CUDA_ARCH_LIST=8.6

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

COPY dockerfiles/rtx3090/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install torch last to override any version pulled by dependencies above
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# flash-attn requires torch installed first and CUDA headers from base image
RUN pip install flash-attn --no-build-isolation

COPY . .

RUN ln -s /usr/bin/python3.10 /usr/bin/python

ENV WANDB_API_KEY=<your_token>

RUN chmod +x src/start.sh
CMD ["bash", "src/start.sh"]
