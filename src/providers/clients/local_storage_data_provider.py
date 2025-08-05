import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas import DataFrame

from src.providers.preprocessors.coinmarketcap_preprocessor import CoinMarketCapPreProcessor
from src.providers.preprocessor import PreProcessor
from src.providers.history_data_provider import HistoryDataProvider


class LocalStorageDataProvider(HistoryDataProvider):
    def __init__(self, directory: str):
        self.directory = Path(os.getcwd()).joinpath(directory)

    def get_ticker_data(
            self, ticker_symbol: str,
            _from_date: Optional[datetime] = None,
            _to_date: Optional[datetime] = None
    ) -> DataFrame:
        file_path = f"{self.directory}/coinmarketcap/history/{ticker_symbol.lower()}-usd.csv"

        if os.path.exists(file_path):
            history = pd.read_csv(file_path, sep=";")
            return history

        raise FileNotFoundError(f"Data source for ticker: {ticker_symbol} does not exist in: {file_path}.")

    def update_ticker_data(self, ticker_symbol: str, market_data: DataFrame) -> DataFrame:
        file_path = f"{self.directory}/coinmarketcap/history/{ticker_symbol.lower()}-usd.csv"

        if os.path.exists(file_path):
            existing_data = pd.read_csv(file_path, sep=";")
            updated_data = pd.concat([market_data, existing_data], ignore_index=True)
            updated_data = updated_data.drop_duplicates()
        else:
            updated_data = market_data

        updated_data.to_csv(file_path, sep=";", index=False)
        return updated_data

    def get_preprocessor(self) -> PreProcessor:
        return CoinMarketCapPreProcessor()
