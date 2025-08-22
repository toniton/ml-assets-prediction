import pandas as pd

from src.entities.asset_entity import AssetEntity
from api.interfaces.market_data import MarketData


class DataframeFactory:
    @staticmethod
    def from_market_data_entity(asset: AssetEntity, market_data: MarketData) -> pd.DataFrame:
        return pd.DataFrame({
            'name': [asset.id],
            'open': [float(market_data.low_price)],
            'high': [float(market_data.high_price)],
            'low': [float(market_data.low_price)],
            'close': [float(market_data.close_price)],
            'volume': [float(market_data.volume)],
            'marketCap': [asset.market_cap],
            'timestamp': [market_data.timestamp]
        })
