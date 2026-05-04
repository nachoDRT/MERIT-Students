# MERIT-Students
Repository for assembling the Merit-Students dataset, available on Hugging Face 🤗. It merges data from the Merit Dataset and images from FairFace.

To generate new student images (WIP) 🛠️:

### Create the docker 🐳
```bash
docker build -f dockerfiles/rtx4090/Dockerfile -t merit-students .
```

### Run 💥 the docker
```bash
docker run -it --gpus device=0 -v "$HOME/.cache/huggingface":/root/.cache/huggingface -v "$(pwd)/src/output":/app/src/output --ipc=host merit-students
```

# Biases Detection - WIP
