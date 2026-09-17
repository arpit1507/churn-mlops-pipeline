from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataTrainingConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_path: Path
    metrics_path: Path
    n_estimators: int
    max_depth: int
    learning_rate: float
    scale_pos_weight: float
    random_state: int