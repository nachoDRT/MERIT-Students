FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# Install Python 3.10
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    libjpeg-dev \
    zlib1g-dev \
    git

# Make Python 3.10 the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Create the working directory
WORKDIR /app

RUN pip3 install torch torchvision torchaudio

RUN pip install git+https://github.com/huggingface/diffusers.git

# Copy and install requirements
COPY dockerfiles/rtx4090/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create symbolic link for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

ENV HUGGINGFACE_HUB_TOKEN=<your_token>

# Execute the training
RUN chmod +x src/start.sh
CMD ["bash", "src/start.sh"]
