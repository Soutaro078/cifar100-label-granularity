from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
CIFAR100_PYTHON_DIR = DATA_ROOT / "cifar-100-python"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def _check_pretrain_mode(mode: str) -> None:
    if mode not in {"fine", "coarse"}:
        raise ValueError(f"mode は 'fine' または 'coarse' です: {mode}")


def pretrain_checkpoint_path(mode: str) -> Path:
    """Fine / Coarse 事前学習の重みを混同しない保存先。

    outputs/models/resnet18_pretrain_fine.pth
    outputs/models/resnet18_pretrain_coarse.pth
    """
    _check_pretrain_mode(mode)
    return MODELS_DIR / f"resnet18_pretrain_{mode}.pth"


def pretrain_resume_path(mode: str) -> Path:
    """最適化の状態も含む途中再開用ファイル。"""
    _check_pretrain_mode(mode)
    return MODELS_DIR / f"resnet18_pretrain_{mode}.resume.pt"


def pretrain_status_path(mode: str) -> Path:
    """画面が止まったとき、別端末から進捗を見るためのテキスト。"""
    _check_pretrain_mode(mode)
    return MODELS_DIR / f"resnet18_pretrain_{mode}.status.txt"
