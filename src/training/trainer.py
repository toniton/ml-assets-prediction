from abc import ABC, abstractmethod
from pathlib import Path


from constants import PROJECT_ROOT
from src.entities.asset_entity import AssetEntity
from src.providers.preprocessor import PreProcessor
from src.providers.history_data_provider import HistoryDataProvider


class Trainer(ABC):
    def __init__(self, model_dir: str, data_provider: HistoryDataProvider, pre_processor: PreProcessor):
        self.model_dir = model_dir
        self.data_provider = data_provider
        self.pre_processor = pre_processor

    def _get_file_path(self, asset: AssetEntity, model_name: str) -> Path:
        file_path = PROJECT_ROOT.joinpath(Path(f"{self.model_dir}/{asset.ticker_symbol.lower()}-{model_name}.joblib"))
        return file_path

    @abstractmethod
    def train_and_save(self, asset: AssetEntity):
        raise NotImplementedError("Trainer method:`train_and_save` has not yet been implemented!")
