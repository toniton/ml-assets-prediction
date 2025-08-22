from __future__ import annotations

import logging
import os.path

from pathlib import Path

import joblib

from api.interfaces.prediction_model import PredictionModel
from constants import PROJECT_ROOT
from src.entities.asset_entity import AssetEntity

logger = logging.getLogger(__name__)


class PredictionModelLoader:
    def __init__(self, prediction_dir: str, cache_dir: str):
        super().__init__()
        self.__directory = Path(prediction_dir)
        self.__cache_dir = Path(cache_dir).expanduser().resolve()
        self.__cache_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}

    def __get_filename(self, ticker_symbol: str, model_class_name: str) -> Path:
        filename = PROJECT_ROOT.joinpath(
            Path(f"{self.__directory}/{ticker_symbol}-{model_class_name}.joblib")
        )
        return filename

    def load_model(self, asset: AssetEntity, model_class_name: str) -> None:
        model_class_name = model_class_name.lower()
        ticker_symbol = asset.ticker_symbol.lower()
        filename = self.__get_filename(ticker_symbol, model_class_name)
        logger.info("Fetching ClassifierModel from file: filepath=%s.", filename)
        if os.path.isfile(filename):
            try:
                logger.info("Loading ClassifierModel from path.")
                if ticker_symbol not in self.models:
                    self.models[ticker_symbol] = {}
                loaded_model = joblib.load(str(filename))
                loaded_model.set_cache_dir(str(self.__cache_dir))
                self.models[ticker_symbol][model_class_name] = loaded_model
                return
            except Exception as exc:
                logger.exception("Failed loading ClassifierModel.")
                raise RuntimeError(["Unable to load model for the requested asset.", asset.name, exc]) from exc
        logger.warning("Model not loaded and may lead to predict failure! Check path: filepath=%s.", filename)
        raise FileNotFoundError(filename)

    def get_model(self, asset: AssetEntity, model_class_name: str) -> PredictionModel:
        model_class_name = model_class_name.lower()
        ticker_symbol = asset.ticker_symbol.lower()
        if ticker_symbol not in self.models:
            self.models[ticker_symbol] = {}
        if model_class_name not in self.models[ticker_symbol]:
            self.load_model(asset, model_class_name)
        return self.models[ticker_symbol][model_class_name]
