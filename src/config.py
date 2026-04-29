from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@dataclass
class CFG:
    data_dir: Path = BASE_DIR / "data" / "plantvillage_small"

    img_size: int = 128
    batch_size: int = 16
    epochs: int = 2
    lr: float = 1e-3

    val_split: float = 0.15
    test_split: float = 0.15
    target_test_size: int = 1000

    seed: int = 42

    runs_dir: Path = BASE_DIR / "runs"

    experiment_name: str = "blur"

    use_noise: bool = False
    noise_std: float = 0.08

    use_blur: bool =True
    blur_kernel: int = 5