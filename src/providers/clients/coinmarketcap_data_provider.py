from datetime import datetime
from typing import Optional

from pandas import DataFrame

from src.providers.preprocessors.coinmarketcap_preprocessor import CoinMarketCapPreProcessor
from src.providers.preprocessors import PreProcessor
from src.providers.history_data_provider import HistoryDataProvider


class CoinMarketCapDataProvider(HistoryDataProvider):

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key

    def get_ticker_data(
        self, ticker_symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ):
        pass

    def update_ticker_data(self, ticker_symbol: str, market_data: DataFrame) -> DataFrame:
        pass

    def get_preprocessor(self) -> PreProcessor:
        return CoinMarketCapPreProcessor()
