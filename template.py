from pathlib import Path
import os
from src.pipeline.logger import logging

list_of_files=[
    "README.md",
    "setup.py",
    "template.py",
    "src/pipeline/logger.py",
    "src/__init__.py",
    "src/pipeline/__init__.py",
    "src/pipeline/config/__init__.py",
    "src/pipeline/config/configuration_manager.py",
    "src/pipeline/data_ingestion/__init__.py",
    "src/pipeline/data_ingestion/data_ingestion.py",
    "src/pipeline/data_ingestion/data_ingestion_config.py",
    "src/pipeline/data_preparation/__init__.py",
    "src/pipeline/data_preparation/data_preparation.py",
    "src/pipeline/data_preparation/data_preparation_config.py",
    "src/pipeline/model_trainer/__init__.py",
    "src/pipeline/model_trainer/model_trainer.py",
    "src/pipeline/model_trainer/model_trainer_config.py",
    "src/pipeline/model_evaluation/__init__.py",
    "src/pipeline/model_evaluation/model_evaluation.py",
    "config/config.yaml",
    "params.yaml",
]

for file in list_of_files:
    file_path = Path(file)
    
    # Check if the file already exists on disk
    if not file_path.exists():
        # Create parent directories only if the file has a parent folder
        if file_path.parent != Path('.'):
            file_path.parent.mkdir(parents=True, exist_ok=True)

        file_path.touch()
        logging.info(f"Created file: {file_path}")
    else:
        logging.info(f"File already exists: {file_path}")


