from __future__ import annotations

import abc
from abc import ABC

from api.interfaces.market_data import MarketData


class PredictionModel(ABC):
    def __init__(self):
        self.model = None

    @abc.abstractmethod
    def load_model(self) -> PredictionModel:
        raise NotImplementedError()

    def is_loaded(self) -> bool:
        return self.model is not None

    @abc.abstractmethod
    def predict(self, current_data: MarketData) -> list[int]:
        raise NotImplementedError()
