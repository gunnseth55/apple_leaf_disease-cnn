from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class LesionConfig:
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    focal_gamma: float = 2.0
    focal_alpha: float = 0.75
    dice_weight: float = 0.5
    focal_weight: float = 0.5
    threshold: float = 0.5
    seed: int = 42
    num_workers: int = 0
    artifacts_dir: Path = Path("artifacts/lesion_detection")

    def to_dict(self):
        values = asdict(self)
        values["artifacts_dir"] = str(self.artifacts_dir)
        return values
