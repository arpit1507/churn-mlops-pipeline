import os
import shutil
from pathlib import Path
import kagglehub
from pipeline.data_ingestion.data_ingestion_config import DataIngestionConfig
from pipeline.config.configuration_manager import ConfigurationManager
from pipeline.logger import logging


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self) -> Path:
        """
        Downloads dataset via kagglehub and syncs it to local artifacts folder.
        """
        target_file = self.config.raw_data_dir / "creditcard.csv"

        if target_file.exists():
            logging.info(f"Dataset already present at {target_file}. Skipping download.")
            return target_file

        logging.info(f"Downloading {self.config.dataset_handle} via kagglehub...")
        cache_path = kagglehub.dataset_download(self.config.dataset_handle)
        logging.info(f"Downloaded to temporary cache: {cache_path}")

        # Locate the CSV inside the downloaded cache folder
        cache_dir = Path(cache_path)
        csv_files = list(cache_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV file found in downloaded path: {cache_path}")

        source_csv = csv_files[0]
        shutil.copy(source_csv, target_file)
        logging.info(f"Copied raw dataset to: {target_file}")

        return target_file
    
if __name__ == "__main__":
    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_data()
