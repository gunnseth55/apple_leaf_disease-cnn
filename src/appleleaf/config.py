from dataclasses import asdict, dataclass
from pathlib import Path


APPLE_CLASSES = (
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
)


@dataclass(frozen=True)
class ExperimentConfig:
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 0
    artifacts_dir: Path = Path("artifacts")

    def to_dict(self) -> dict:
        values = asdict(self)
        values["artifacts_dir"] = str(self.artifacts_dir)
        return values

