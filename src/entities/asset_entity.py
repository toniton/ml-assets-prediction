#!/usr/bin/env python3
from pydantic.dataclasses import dataclass


@dataclass
class AssetEntity:
    keywords: list[str]
    ticker_symbol: str
    decimal_places: int
    name: str
    id: float
    exchange: str
    market_cap: str
