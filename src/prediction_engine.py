from __future__ import annotations
import logging

from api import PredictionModelLoader
from api.interfaces.market_data import MarketData
from src.entities.asset_entity import AssetEntity
from src.providers.history_data_provider import HistoryDataProvider
from src.training.random_forest.random_forest_classifier_model import RandomForestClassifierModel
from src.training.random_forest.random_forest_classifier_trainer import RandomForestClassifierTrainer


class PredictionEngine:
    def __init__(
            self, assets: list[AssetEntity], data_provider: HistoryDataProvider,
            prediction_dir: str, prediction_model_loader: PredictionModelLoader
    ):
        self.asset_lookup: dict[str, AssetEntity] = {}
        self.prediction_model_loader = prediction_model_loader
        self.data_provider: HistoryDataProvider = data_provider
        self.assets = assets
        self.prediction_dir = prediction_dir
        self.init_application()

    def init_application(self):
        self.load_models()
        for asset in self.assets:
            self.asset_lookup[asset.ticker_symbol] = asset

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

    def load_models(self) -> None:
        for asset in self.assets:
            try:
                self.prediction_model_loader.load_model(asset, str(RandomForestClassifierModel.__name__).lower())
            except Exception as exc:
                logging.error(["Error occurred loading model. ->", exc])

    def predict(self, ticker_symbol: str, current_data: MarketData) -> int:
        asset = self.asset_lookup.get(ticker_symbol.lower())
        if not asset:
            raise ValueError(f"Asset with ticker symbol {ticker_symbol} not found in prediction engine.")
        prediction_model = self.prediction_model_loader.get_model(
            asset, str(RandomForestClassifierModel.__name__).lower()
        )
        if prediction_model:
            predictions = prediction_model.predict([current_data])
            return predictions[0]

        raise ValueError(
            f"Model for {ticker_symbol} ticker not found in prediction engine. "
            f"Load assets model and try again."
        )
