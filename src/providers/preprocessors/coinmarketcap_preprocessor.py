import logging

import numpy as np
from pandas import DataFrame

from src.helpers.dataframe_helper import DataFrameHelper
from src.providers.preprocessor import PreProcessor


class CoinMarketCapPreProcessor(PreProcessor):
    horizons = [10, 50, 100, 250, 500, 1000]

    @classmethod
    def get_horizon(
            cls, training_data: DataFrame, predictors: list[str]
    ) -> tuple[DataFrame, list]:
        for horizon in cls.horizons:
            if horizon < training_data.shape[0]:
                rolling_averages = (
                    training_data["close"].rolling(window=horizon, min_periods=1).mean()
                )
                rolling_sums = (
                    training_data["target"].rolling(window=horizon, min_periods=1).sum()
                )

                ratio_column = f"close_Ratio_{horizon}"
                training_data[ratio_column] = training_data["close"] / rolling_averages

                trend_column = f"trend_{horizon}"
                training_data[trend_column] = rolling_sums
                # = .shift(1)

                predictors.extend([ratio_column, trend_column])
            else:
                logging.warning("Horizon %s exceeds data size.", horizon)
        return training_data, predictors

    def pre_process_data(
            self, data: DataFrame
    ) -> tuple[DataFrame, list[str], DataFrame]:
        DataFrameHelper.normalize_timestamp(data)
        data["tomorrow"] = data["close"].shift(-1)
        data["target"] = (
                (data["tomorrow"] > data["close"]) & data["tomorrow"].notna()
        ).astype(int)

        data["ema_10"] = data["close"].ewm(span=10, adjust=False).mean()
        data["ema_20"] = data["close"].ewm(span=20, adjust=False).mean()
        data["rsi_14"] = 100 - (100 / (1 + (data["close"].diff().clip(lower=0).rolling(14).mean() /
                                            (-data["close"].diff().clip(upper=0).rolling(14).mean()))))
        data["return"] = data["close"].pct_change()
        data["log_return"] = np.log(data["close"] / data["close"].shift(1))

        # Trend
        for w in (3, 6, 12):
            data[f"sma_{w}"] = data["close"].rolling(w).mean()
        data["price_sma6"] = data["close"] / data["sma_6"]
        data["sma3_sma12"] = data["sma_3"] / data["sma_12"]

        # Momentum
        data["mom_3"] = data["close"].pct_change(3)
        data["mom_6"] = data["close"].pct_change(6)
        data["mom_12"] = data["close"].pct_change(12)

        predictors_copy = [
            col for col in data.columns
            if col not in {"timeOpen", "timeClose", "timeHigh", "timeLow", "marketCap", "tomorrow", "target"}
        ]
        clean_data, predictors = self.get_horizon(data, predictors_copy)
        target = clean_data.target
        print(predictors)
        return clean_data, predictors, target
