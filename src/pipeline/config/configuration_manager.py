from pathlib import Path
import yaml
from pipeline.data_ingestion.data_ingestion_config import DataIngestionConfig
from pipeline.data_preparation.data_preparation_config import DataPreparationConfig
from pipeline.model_training.model_trainer_config import DataTrainingConfig

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

    def get_data_preparation_config(self) -> DataPreparationConfig:
        cfg = self.config["data_preparation"]
        params = self.params["data_preparation"]
        Path(cfg["root_dir"]).mkdir(parents=True, exist_ok=True)

        return DataPreparationConfig(
            root_dir=Path(cfg["root_dir"]),
            raw_data_path=Path(cfg["raw_data_path"]),
            train_data_path=Path(cfg["train_data_path"]),
            test_data_path=Path(cfg["test_data_path"]),
            preprocessor_path=Path(cfg["preprocessor_path"]),
            test_size=params["test_size"],
            random_state=params["random_state"],
            enable_resampling=params["enable_resampling"]
        )

    def get_data_training_config(self) -> DataTrainingConfig:
        config = self.config["model_training"]
        params = self.params["model_training"]

        Path(config["root_dir"]).mkdir(parents=True, exist_ok=True)

        return DataTrainingConfig(
            root_dir=Path(config["root_dir"]),
            train_data_path=Path(config["train_data_path"]),
            test_data_path=Path(config["test_data_path"]),
            model_path=Path(config["model_path"]),
            metrics_path=Path(config["metrics_path"]),
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=params["scale_pos_weight"],
            random_state=params["random_state"]
        )