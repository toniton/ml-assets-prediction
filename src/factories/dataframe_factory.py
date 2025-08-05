from datetime import datetime

import pandas as pd

from src.entities.asset_entity import AssetEntity
from api.interfaces.market_data import MarketData


class DataframeFactory:
    @staticmethod
    def from_market_data_entity(asset: AssetEntity, market_data: MarketData) -> pd.DataFrame:
        _date_utc_now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        return pd.DataFrame({
            'timeOpen': [market_data.timestamp],
            'timeClose': [market_data.timestamp],
            'timeHigh': [market_data.timestamp],
            'timeLow': [market_data.timestamp],
            # 'name': [asset.name],
            'open': [market_data.low_price],
            'high': [market_data.high_price],
            'low': [market_data.low_price],
            'close': [market_data.close_price],
            'volume': [market_data.volume],
            'marketCap': [asset.market_cap],
            # 'date': [market_data.timestamp]
        })
