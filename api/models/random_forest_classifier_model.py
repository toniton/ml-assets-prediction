from __future__ import annotations

import logging
import os.path
from abc import ABC
from pathlib import Path
from typing import Self

import joblib
from sklearn.ensemble import RandomForestClassifier

from api.interfaces.prediction_model import PredictionModel
from api.interfaces.market_data import MarketData
from src.entities.asset_entity import AssetEntity
from src.factories.dataframe_factory import DataframeFactory

logger = logging.getLogger(__name__)


class RandomForestClassifierModel(PredictionModel, ABC):
    def __init__(self, asset: AssetEntity, prediction_dir: str):
        super().__init__()
        self.asset = asset
        self.directory = Path(prediction_dir)
        self.model: RandomForestClassifier | None = None

    def get_filename(self) -> Path:
        filename = Path(f"{self.directory}/{self.asset.ticker_symbol.lower()}-random-forest.joblib")
        return filename

    def load_model(self) -> Self:
        filename = self.get_filename()
        logger.info(f"Fetching RandomForestClassifierModel from file: filepath={filename}.")
        if os.path.isfile(filename):
            try:
                logger.info(f"Loading RandomForestClassifierModel from path.")
                self.model = joblib.load(filename)
                return self
            except Exception as exc:
                logger.exception(f"Failed loading RandomForestClassifierModel.")
                raise RuntimeError(["Unable to load model for the requested asset.", self.asset.name, exc])
        logger.warning("Model not loaded and may lead to predict failure! Check path: filepath=%s.", filename)
        raise NotImplementedError("# TODO: Download model from artifactory.")

    def predict(self, current_data: MarketData) -> list[int]:
        if self.model:
            data = DataframeFactory.from_market_data_entity(self.asset, current_data)
            return self.model.predict(data)
        logger.exception("Prediction failure! Could not find RandomForestClassifierModel.")
        raise RuntimeError(["You need to load or train model before prediction."])
