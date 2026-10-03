#!/usr/bin/env bash
# ホストの src / outputs / data / configs をコンテナへマウントして実行する。
# 引数が無いときは GPU が見えるかだけ確認する。
# 例: ./scripts/docker-run.sh python src/pretrain.py --mode fine --no_wandb
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-cifar100-label-granularity:cu124}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  docker build -t "$IMAGE" "$ROOT"
fi

args=(
  --rm
  --gpus all
  --shm-size=8g
  -e MPLBACKEND=Agg
  -e MKL_THREADING_LAYER=GNU
  -e PYTHONUNBUFFERED=1
  -w /workspace
  -v "$ROOT/src:/workspace/src"
  -v "$ROOT/outputs:/workspace/outputs"
  -v "$ROOT/data:/workspace/data"
  -v "$ROOT/configs:/workspace/configs"
)

if [[ -f "$HOME/.netrc" ]]; then
  args+=(-v "$HOME/.netrc:/root/.netrc:ro")
fi

if [[ -t 0 && -t 1 ]]; then
  args+=(-it)
fi

if [[ $# -eq 0 ]]; then
  set -- python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
fi

exec docker run "${args[@]}" "$IMAGE" "$@"
