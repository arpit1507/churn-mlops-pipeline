from pipeline.logger import logging
from pipeline.data_ingestion.data_ingestion import DataIngestion
from pipeline.data_preparation.data_preparation import DataPreparation
from pipeline.data_ingestion.data_ingestion_config import DataIngestionConfig
from pipeline.data_preparation.data_preparation_config import DataPreparationConfig

if __name__ == "__main__":
    # Initialize configuration manager
    from pipeline.config.configuration_manager import ConfigurationManager
    config_manager = ConfigurationManager()

    # Data Ingestion
    data_ingestion_config: DataIngestionConfig = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    raw_data_path = data_ingestion.download_data()
    logging.info(f"Raw data downloaded to: {raw_data_path}")

    # Data Preparation
    data_preparation_config: DataPreparationConfig = config_manager.get_data_preparation_config()
    data_preparation = DataPreparation(config=data_preparation_config)
    train_data_path, test_data_path = data_preparation.prepare_data(raw_data_path)
    logging.info(f"Data prepared. Train data at: {train_data_path}, Test data at: {test_data_path}")