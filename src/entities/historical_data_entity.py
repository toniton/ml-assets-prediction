#!/usr/bin/env python3
from pydantic import BaseModel

from src.entities.asset_entity import AssetEntity


class HistoricalData(BaseModel):
    asset: AssetEntity
    year_week: str
    timestamp: str
    open_price: float
    close_price: float
    high_price: float
    low_price: float
    volume: int
