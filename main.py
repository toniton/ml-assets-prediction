from api import PredictionModelLoader
from configuration.assets_config import AssetsConfig
from src.prediction_engine import PredictionEngine
from src.providers.clients.local_storage_data_provider import LocalStorageDataProvider


def main():
    assets_config = AssetsConfig()
    assets = assets_config.assets

    local_storage_data_provider = LocalStorageDataProvider(directory='./storage/datasets')
    prediction_model_loader = PredictionModelLoader('./tests/models', './tests/cache')
    prediction_engine = PredictionEngine(assets, local_storage_data_provider, 'storage/models', prediction_model_loader)
    prediction_engine.train_assets_model()


if __name__ == "__main__":
    main()
