from pathlib import Path
import yaml
import pytest
from pipeline.config.configuration_manager import ConfigurationManager

def test_yaml_files_exist():
    assert Path("config/config.yaml").exists(), "config.yaml is missing"
    assert Path("params.yaml").exists(), "params.yaml is missing"

def test_config_keys():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    assert "data_ingestion" in config
    assert "data_preparation" in config
    assert "model_training" in config

def test_configuration_manager_initialization():
    manager = ConfigurationManager()
    ingestion_cfg = manager.get_data_ingestion_config()
    prep_cfg = manager.get_data_preparation_config()
    train_cfg = manager.get_data_training_config()

    assert ingestion_cfg is not None
    assert prep_cfg is not None
    assert train_cfg is not None