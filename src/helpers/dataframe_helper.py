import pandas as pd


class DataFrameHelper:
    @staticmethod
    def normalize_timestamp(data: pd.DataFrame) -> pd.DataFrame:
        data["timestamp"] = pd.to_numeric(
            data["timestamp"], errors="coerce"
        ).fillna(
            pd.to_datetime(data["timestamp"], utc=True).astype("int64") // 10 ** 9
        ).astype("int64")
        return data
