from __future__ import annotations
import logging
from pandas import DataFrame

from api import PredictionModel, RandomForestClassifierModel
from src.entities.asset_entity import AssetEntity
from api.interfaces.market_data import MarketData
from src.factories.dataframe_factory import DataframeFactory
from src.providers.history_data_provider import HistoryDataProvider
from src.training.models.random_forest_classifier_trainer import RandomForestClassifierTrainer


class PredictionEngine:
    def __init__(self, assets: list[AssetEntity], data_provider: HistoryDataProvider, prediction_dir: str):
        self.models: dict[str, PredictionModel] = {}
        self.history_cache: dict[str, DataFrame] = {}
        self.asset_lookup: dict[str, AssetEntity] = {}
        self.data_provider: HistoryDataProvider = data_provider
        self.assets = assets
        self.prediction_dir = prediction_dir
        self.init_application()

    def init_application(self):
        self.load_models()
        for asset in self.assets:
            try:
                self.asset_lookup[asset.ticker_symbol] = asset
                data = self.data_provider.get_ticker_data(asset.ticker_symbol)
                self.history_cache[asset.ticker_symbol] = data.head(1200).copy()
            except Exception as exc:
                logging.error(["Error occurred initializing application. ->", exc])

    # def start_training(self):
    #     try:
    #         schedule.every().week.do(self.train_assets_model())
    #         # schedule.every().day.do(self.update_data_source(assets))
    #     except Exception as exc:
    #         logging.error(["Error occurred initializing application. ->", exc])

    def set_data_provider(self, data_provider: HistoryDataProvider):
        self.data_provider = data_provider

    def train_assets_model(self) -> PredictionEngine:
        for asset in self.assets:
            preprocessor = self.data_provider.get_preprocessor()
            trainer = RandomForestClassifierTrainer(
                self.prediction_dir, self.data_provider, preprocessor
            )
            trainer.train_and_save(asset)
        return self

    def load_model(self, ticker_symbol: str) -> PredictionModel:
        if ticker_symbol in self.models:
            model = self.models[ticker_symbol]
            if model.model is None:
                model.load_model()
            return model
        raise ValueError(
            f"Model for {ticker_symbol} ticker not found in prediction engine. "
            f"Load assets model and try again."
        )

    def load_models(self):
        for asset in self.assets:
            try:
                model = RandomForestClassifierModel(asset, self.prediction_dir)
                model.load_model()
                self.models[asset.ticker_symbol] = model
            except Exception as exc:
                logging.error(["Error occurred loading model. ->", exc])
        return self.models

    def predict(self, ticker_symbol: str, current_data: MarketData) -> int:
        prediction_model = self.models.get(ticker_symbol)
        if prediction_model:
            predictions = prediction_model.predict(current_data)
            return predictions[0]

        raise ValueError(
            f"Model for {ticker_symbol} ticker not found in prediction engine. "
            f"Load assets model and try again."
        )

    def fine_tune_model(self, ticker_symbol: str, market_data: MarketData):
        preprocessor = self.data_provider.get_preprocessor()
        random_forest_classifier_trainer = RandomForestClassifierTrainer(
            self.prediction_dir, self.data_provider, preprocessor
        )
        asset = self.asset_lookup.get(ticker_symbol)
        recent_data = DataframeFactory.from_market_data_entity(asset, market_data)
        random_forest_classifier_trainer.fine_tune_model(asset, recent_data)
