import abc
from datetime import datetime
from typing import Optional

from pandas import DataFrame

from src.providers.preprocessor import PreProcessor


class HistoryDataProvider(abc.ABC):
    @abc.abstractmethod
    def get_ticker_data(
            self,
            ticker_symbol: str,
            from_date: Optional[datetime] = None,
            to_date: Optional[datetime] = None
    ) -> DataFrame:
        raise NotImplementedError()

    @abc.abstractmethod
    def update_ticker_data(
            self,
            ticker_symbol: str,
            market_data: DataFrame
    ) -> DataFrame:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_preprocessor(self) -> PreProcessor:
        raise NotImplementedError()
