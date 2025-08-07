import logging

import pandas as pd
from pandas import DataFrame

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
            training_data['ema_20'] = training_data['close'].ewm(span=20, adjust=False).mean()
        return training_data, predictors

    def pre_process_data(
            self, data: DataFrame
    ) -> tuple[DataFrame, list[str], DataFrame]:
        data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
        data["timestamp"] = data["timestamp"].astype("int64") // 10 ** 9
        print(data["timestamp"])
        # data["date"] = data["timestamp"].astype('int64') // 10 ** 9
        data["tomorrow"] = data["close"].shift(-1)
        data["target"] = (data["tomorrow"] > data["close"]).apply(int)
        clean_data = data.drop(data.index[-1])

        predictors_copy = list(data.columns)
        predictors_copy.remove("timeOpen")
        predictors_copy.remove("timeClose")
        predictors_copy.remove("timeHigh")
        predictors_copy.remove("timeLow")
        clean_data, predictors = self.get_horizon(clean_data, predictors_copy)
        target = clean_data.target
        print(predictors)
        return clean_data, predictors, target
