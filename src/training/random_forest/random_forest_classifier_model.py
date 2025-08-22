from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier

from api.interfaces.market_data import MarketData
from api.interfaces.prediction_model import PredictionModel
from src.entities.asset_entity import AssetEntity
from src.factories.dataframe_factory import DataframeFactory
from src.helpers.dataframe_helper import DataFrameHelper
from src.providers.preprocessor import PreProcessor

logger = logging.getLogger(__name__)


class RandomForestClassifierModel(PredictionModel[RandomForestClassifier]):
    MAX_WINDOW_SIZE = 1000

    @property
    def feature_names(self) -> list[str]:
        return self.__feature_names

    @property
    def training_subset(self) -> DataFrame:
        return self.__training_subset[self.feature_names]

    @property
    def model(self) -> RandomForestClassifier | None:
        if self.__model is None:
            logger.error("Model is not loaded. Please load the model before accessing it.")
            raise RuntimeError("Model is not loaded. Please load the model before accessing it.")
        logger.debug("Returning RandomForestClassifierModel: %s", self.__model)
        return self.__model

    def __init__(
            self, model: RandomForestClassifier, feature_names: list[str],
            training_subset: DataFrame, asset: AssetEntity, preprocessor: PreProcessor
    ):
        super().__init__()
        self.__cache_dir = None
        self.__cache_file = None
        self.__model = model
        self.__feature_names = feature_names
        self.__training_subset = training_subset
        self.__preprocessor = preprocessor
        self.asset = asset

    def set_cache_dir(self, cache_dir: str):
        self.__cache_dir = Path(cache_dir).expanduser().resolve()
        self.__cache_dir.mkdir(parents=True, exist_ok=True)
        self.__cache_file = self.__cache_dir / f"{self.asset.ticker_symbol.lower()}-train-subset.csv"

        if self.__cache_file.is_file():
            try:
                logger.info("Loading training subset cache from %s", self.__cache_file)
                self.__training_subset = pd.read_csv(
                    self.__cache_file, dtype={"timestamp": int}
                )
                DataFrameHelper.normalize_timestamp(self.__training_subset)
                return
            except Exception as e:
                logger.warning("Failed to load cache file %s: %s", self.__cache_file, e)
        self.__save_cache()

    def __save_cache(self):
        if self.__cache_file:
            self.__training_subset.to_csv(self.__cache_file, index=False)
            logger.debug("Training subset saved to %s", self.__cache_file)

    def __update_cache_with_market_data(self, new_data: DataFrame):
        self.__training_subset = new_data

        if len(self.__training_subset) > self.MAX_WINDOW_SIZE:
            self.__training_subset = self.__training_subset.iloc[-self.MAX_WINDOW_SIZE:]

        self.__save_cache()

    def predict(self, current_data: list[MarketData], update: bool = True) -> list[int]:
        if self.model:
            data = DataframeFactory.from_market_data_entity(self.asset, current_data[0])
            new_data = pd.concat(
                [data, self.__training_subset],
                ignore_index=False
            )
            pre_processed_data, _, _ = self.__preprocessor.pre_process_data(new_data)
            filtered_data = pre_processed_data[self.__model.feature_names_in_]
            if update:
                self.fine_tune(filtered_data)

            selected = filtered_data.tail(1)
            return self.model.predict(selected)
        logger.exception("Prediction failure! Could not find RandomForestClassifierModel.")
        raise RuntimeError("You need to load or train model before prediction.")

    def fine_tune(self, update_data: DataFrame):
        logger.info("Fine-tuning model for asset: name=%s.", self.asset.name)
        self.__update_cache_with_market_data(update_data)
