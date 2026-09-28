from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"設定ファイルが辞書形式ではありません: {path}")
    return data


def apply_overrides(config: dict, overrides: dict) -> dict:
    """None でない CLI 引数だけ設定を上書きする。"""
    merged = dict(config)
    for key, value in overrides.items():
        if value is not None:
            merged[key] = value
    return merged
