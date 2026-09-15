from pathlib import Path
import yaml
from pipeline.data_ingestion.data_ingestion_config import DataIngestionConfig
from pipeline.data_preparation.data_preparation_config import DataPreparationConfig
from pipeline.model_trainer.model_trainer_config import ModelTrainerConfig

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = Path("config/config.yaml"),
        params_filepath: Path = Path("params.yaml")
    ):
        with open(config_filepath, "r") as f:
            self.config = yaml.safe_load(f)
        with open(params_filepath, "r") as f:
            self.params = yaml.safe_load(f)

        Path(self.config["artifacts_root"]).mkdir(parents=True, exist_ok=True)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        cfg = self.config["data_ingestion"]
        Path(cfg["root_dir"]).mkdir(parents=True, exist_ok=True)
        Path(cfg["raw_data_dir"]).mkdir(parents=True, exist_ok=True)

        return DataIngestionConfig(
            root_dir=Path(cfg["root_dir"]),
            dataset_handle=cfg["dataset_handle"],
            raw_data_dir=Path(cfg["raw_data_dir"])
        )

