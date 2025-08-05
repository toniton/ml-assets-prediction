import logging

import joblib
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from src.entities.asset_entity import AssetEntity
from src.helpers.random_forest_classifier_helper import RandomForestClassifierHelper
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class RandomForestClassifierTrainer(Trainer):
    def __train_model(self, historical_data: DataFrame) -> RandomForestClassifier:
        (processed_data, predictors, target) = self.pre_processor.pre_process_data(historical_data)
        filtered_data = processed_data[predictors].values
        helper = RandomForestClassifierHelper()
        return helper.train_model(filtered_data, target)

    def __save_model(self, asset: AssetEntity, model: RandomForestClassifier) -> None:
        filename = self.get_filename(asset, "random-forest")
        logger.info("Saving trained model to path: filepath=%s.", filename)
        joblib.dump(model, filename)

    def train_and_save(self, asset: AssetEntity):
        logger.info("Training model for asset: name=%s.", asset.name)
        historical_data = self.data_provider.get_ticker_data(asset.ticker_symbol)
        trained_model = self.__train_model(historical_data)
        self.__save_model(asset, trained_model)

    def fine_tune_model(self, asset: AssetEntity, recent_data: DataFrame):
        logger.info("Fine-tuning model for asset: name=%s.", asset.name)
        historical_data = self.data_provider.update_ticker_data(asset.ticker_symbol, recent_data)
        logger.warning("Fine-tuning this model will re-train on %s records.", len(historical_data))
        trained_model = self.__train_model(historical_data)
        self.__save_model(asset, trained_model)
