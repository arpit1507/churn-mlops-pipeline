import sys
from pathlib import Path

# Add project root and 'src' directory to Python search path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

# Clean module imports
from pipeline.logger import logging
from pipeline.config.configuration_manager import ConfigurationManager
from pipeline.data_ingestion.data_ingestion import DataIngestion
from pipeline.data_ingestion.data_ingestion_config import DataIngestionConfig
from pipeline.data_preparation.data_preparation import DataPreparation
from pipeline.data_preparation.data_preparation_config import DataPreparationConfig
from pipeline.model_training.model_trainer import DataTraining
from pipeline.model_training.model_trainer_config import DataTrainingConfig

if __name__ == "__main__":
    # Initialize configuration manager
    config_manager = ConfigurationManager()

    # Data Ingestion
    data_ingestion_config: DataIngestionConfig = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    raw_data_path = data_ingestion.download_data()
    logging.info(f"Raw data downloaded to: {raw_data_path}")

    # Data Preparation
    data_preparation_config: DataPreparationConfig = config_manager.get_data_preparation_config()
    data_preparation = DataPreparation(config=data_preparation_config)
    train_data_path, test_data_path = data_preparation.prepare_data()
    logging.info(f"Data prepared. Train data at: {train_data_path}, Test data at: {test_data_path}")

    # Model Training
    data_training_config: DataTrainingConfig = config_manager.get_data_training_config()
    data_training = DataTraining(config=data_training_config)
    data_training.train()
    