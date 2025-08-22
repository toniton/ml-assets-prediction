import logging

import joblib
from pandas import DataFrame

from src.entities.asset_entity import AssetEntity
from src.helpers.random_forest_classifier_helper import RandomForestClassifierHelper
from src.training.random_forest.random_forest_classifier_model import RandomForestClassifierModel
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class RandomForestClassifierTrainer(Trainer):
    def __train_model(self, asset: AssetEntity, historical_data: DataFrame) -> RandomForestClassifierModel:
        (processed_data, predictors, target) = self.pre_processor.pre_process_data(historical_data)
        filtered_data = processed_data[predictors]
        helper = RandomForestClassifierHelper()
        trained_model = helper.train_model(filtered_data, target)
        classifier_model = RandomForestClassifierModel(
            trained_model, predictors, filtered_data,
            asset, self.pre_processor
        )
        return classifier_model

    def __save_model(self, asset: AssetEntity, model: RandomForestClassifierModel) -> None:
        model_class_name = model.__class__.__name__.lower()
        filename = self._get_file_path(asset, model_class_name)
        logger.info("Saving trained model to path: filepath=%s.", filename)
        joblib.dump(model, filename)

    def train_and_save(self, asset: AssetEntity):
        logger.info("Training model for asset: name=%s.", asset.name)
        historical_data = self.data_provider.get_ticker_data(asset.ticker_symbol)
        classifier_model = self.__train_model(asset, historical_data)
        self.__save_model(asset, classifier_model)
