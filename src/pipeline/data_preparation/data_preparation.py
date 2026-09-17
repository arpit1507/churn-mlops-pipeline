from pipeline.data_preparation.data_preparation_config import DataPreparationConfig
from pipeline.logger import logging
from pipeline.config.configuration_manager import ConfigurationManager
import pandas as pd
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,RobustScaler

class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config

    def prepare_data(self):
        """
        Placeholder for data preparation logic.
        """
        logging.info("Starting data preparation...")
        # Implement your data preparation logic here
        df=pd.read_csv(self.config.raw_data_path)
        logging.info("Transforming cyclical Time features")
        seconds_in_day = 24 * 60 * 60
        df["sin_time"] = np.sin(2 * np.pi * (df["Time"] % seconds_in_day) / seconds_in_day)
        df["cos_time"] = np.cos(2 * np.pi * (df["Time"] % seconds_in_day) / seconds_in_day)
        df.drop(columns=["Time"], inplace=True)

        # 2. Stratified Train-Test Split (leakage prevention)
        logging.info(f"Executing stratified split with test_size={self.config.test_size}")
        X = df.drop(columns=["Class"])
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            stratify=y,
            random_state=self.config.random_state
        )

        # 3. Robust Scaling on Amount (fitted exclusively on training split)
        logging.info("Fitting RobustScaler on X_train['Amount']")
        scaler = RobustScaler()
        X_train["scaled_amount"] = scaler.fit_transform(X_train[["Amount"]])
        X_test["scaled_amount"] = scaler.transform(X_test[["Amount"]])
        X_train.drop(columns=["Amount"], inplace=True)
        X_test.drop(columns=["Amount"], inplace=True)

        # Persist scaler pipeline artifact
        joblib.dump(scaler, self.config.preprocessor_path)
        logging.info(f"Saved preprocessor to {self.config.preprocessor_path}")

        # 4. Optional SMOTE (applied strictly on X_train)
        if self.config.enable_resampling:
            logging.info("Applying BorderlineSMOTE resampling to training set only")
            smote = BorderlineSMOTE(random_state=self.config.random_state)
            X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        else:
            X_train_res, y_train_res = X_train, y_train

        # 5. Export as Parquet
        train_df = pd.concat([X_train_res, y_train_res], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_df.to_parquet(self.config.train_data_path, index=False)
        test_df.to_parquet(self.config.test_data_path, index=False)
        logging.info(f"Successfully saved train ({train_df.shape}) and test ({test_df.shape}) parquet files.")
        logging.info("Data preparation completed.")
        return self.config.train_data_path,self.config.test_data_path


if __name__ == "__main__":
    config_manager = ConfigurationManager()
    preparation_config = config_manager.get_data_preparation_config()
    data_preparation = DataPreparation(config=preparation_config)
    data_preparation.prepare_data()