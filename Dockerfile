# ホストの Anaconda (base) は使わない。
# Python 3.11 の公式イメージに、CUDA 12.4 用の PyTorch wheel を入れる。
# GPU を使うとき、コンテナ内の CUDA ライブラリとホストの NVIDIA ドライバを
# nvidia-container-toolkit がつなぐ。
FROM python:3.11.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MKL_THREADING_LAYER=GNU

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# torch は PyPI ではなく CUDA 12.4 の公式 wheel から固定する。
# 先に入れることで、後段の pip が CPU 版 torch を引き込まないようにする。
RUN pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir \
        matplotlib==3.11.2 \
        PyYAML==6.0.3 \
        wandb==0.30.0

COPY configs /workspace/configs
COPY src /workspace/src
COPY requirements.txt README.md /workspace/

RUN mkdir -p /workspace/data /workspace/outputs/models /workspace/outputs/figures

CMD ["python", "-c", "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"]
