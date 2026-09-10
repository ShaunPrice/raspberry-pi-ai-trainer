FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch
RUN python -m pip install --no-cache-dir numpy pillow onnx transformers peft safetensors opencv-python-headless
WORKDIR /workspace
