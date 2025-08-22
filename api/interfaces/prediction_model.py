from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pandas import DataFrame

from api.interfaces.market_data import MarketData

T = TypeVar('T')


class PredictionModel(ABC, Generic[T]):
    @property
    @abstractmethod
    def model(self) -> T:
        pass

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def training_subset(self) -> DataFrame:
        pass

    @abstractmethod
    def predict(self, current_data: list[MarketData], update: bool = True):
        pass

    @abstractmethod
    def fine_tune(self, update_data: DataFrame):
        raise NotImplementedError("PredictionModel method:`train_and_save` has not yet been implemented!")

    @abstractmethod
    def set_cache_dir(self, cache_dir: str):
        raise NotImplementedError("PredictionModel method:`set_cache_dir` has not yet been implemented!")
