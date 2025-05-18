import logging

import joblib
import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from api.interfaces.market_data import MarketData
from src.entities.asset_entity import AssetEntity
from src.helpers.random_forest_classifier_helper import RandomForestClassifierHelper
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class RandomForestClassifierTrainer(Trainer):
    def train_model(self, historical_data: DataFrame) -> RandomForestClassifier:
        (processed_data, predictors, target) = self.pre_processor.pre_process_data(historical_data)
        filtered_data = processed_data[predictors].values
        helper = RandomForestClassifierHelper()
        return helper.train_model(filtered_data, target)

    def save_model(self, asset: AssetEntity, model: RandomForestClassifier) -> None:
        filename = self.get_filename(asset, "random-forest")
        logger.info(f"Saving trained model to path: filepath={filename}.")
        joblib.dump(model, filename)

    def train_and_save(self, asset: AssetEntity):
        logger.info(f"Training model for asset: name={asset.name}.")
        historical_data = self.data_provider.get_ticker_data(asset.ticker_symbol)
        trained_model = self.train_model(historical_data)
        self.save_model(asset, trained_model)

    def fine_tune_model(self, asset: AssetEntity, recent_data: DataFrame):
        logger.info(f"Fine-tuning model for asset: name={asset.name}.")
        historical_data = self.data_provider.update_ticker_data(asset.ticker_symbol, recent_data)
        logger.warning(f"Fine-tuning this model will re-train on {len(historical_data)} records.")
        trained_model = self.train_model(historical_data)
        self.save_model(asset, trained_model)
