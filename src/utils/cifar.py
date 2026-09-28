import pickle
from pathlib import Path

from utils.paths import CIFAR100_PYTHON_DIR


def load_cifar100_meta(data_dir: Path | None = None) -> dict:
    data_dir = data_dir or CIFAR100_PYTHON_DIR
    with open(data_dir / "meta", "rb") as f:
        return pickle.load(f, encoding="latin1")


def load_split_dict(split: str, data_dir: Path | None = None) -> dict:
    if split not in {"train", "test"}:
        raise ValueError(f"split は 'train' または 'test' です: {split}")
    data_dir = data_dir or CIFAR100_PYTHON_DIR
    with open(data_dir / split, "rb") as f:
        return pickle.load(f, encoding="latin1")
