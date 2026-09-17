import json
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier

from pipeline.config.configuration_manager import ConfigurationManager
from pipeline.model_training.model_trainer_config import DataTrainingConfig
from pipeline.logger import logging

class DataTraining:
    def __init__(self, config: DataTrainingConfig):
        self.config = config

    def train(self):
        logging.info(f"Loading training data from {self.config.train_data_path}")
        train_df = pd.read_parquet(self.config.train_data_path)
        test_df = pd.read_parquet(self.config.test_data_path)

        X_train = train_df.drop(columns=["Class"])
        y_train = train_df["Class"]
        X_test = test_df.drop(columns=["Class"])
        y_test = test_df["Class"]

        logging.info(f"Training shapes: X_train={X_train.shape}, X_test={X_test.shape}")

        model = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            scale_pos_weight=self.config.scale_pos_weight,
            random_state=self.config.random_state,
            eval_metric="logloss",
            n_jobs=-1
        )

        logging.info("Fitting model...")
        model.fit(X_train, y_train)

        # Save model artifact
        joblib.dump(model, self.config.model_path)
        logging.info(f"Model saved at: {self.config.model_path}")

        # Baseline evaluation metrics
        y_probs = model.predict_proba(X_test)[:, 1]
        y_preds = model.predict(X_test)

        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_probs)),
            "pr_auc": float(average_precision_score(y_test, y_probs)),
            "f1_score": float(f1_score(y_test, y_preds))
        }

        with open(self.config.metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)

        logging.info(f"Training metrics saved at {self.config.metrics_path}: {metrics}")

if __name__ == "__main__":
    config_manager = ConfigurationManager()
    training_config = config_manager.get_data_training_config()
    trainer = DataTraining(config=training_config)
    trainer.train()