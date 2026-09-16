from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataPreparationConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    preprocessor_path: Path
    raw_data_path: Path
    test_size: int
    random_state: int
    enable_resampling: bool