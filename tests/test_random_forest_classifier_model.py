import os
import tempfile
from datetime import datetime

import numpy
import pytest

from api import PredictionModelLoader
from api.interfaces.market_data import MarketData
from src.entities.asset_entity import AssetEntity
from src.providers.clients.local_storage_data_provider import LocalStorageDataProvider
from src.providers.preprocessors.coinmarketcap_preprocessor import CoinMarketCapPreProcessor
from src.training.random_forest.random_forest_classifier_trainer import RandomForestClassifierTrainer


def test_model_prediction():
    asset = AssetEntity(
        id=2781, name='BTC', ticker_symbol='BTC',
        exchange='CRYPTO_DOT_COM', market_cap='525885640459.76',
        decimal_places=8, keywords=[]
    )
    model_class_name = "randomforestclassifiermodel"
    loader = PredictionModelLoader('./tests/models', './tests/cache')
    loader.load_model(asset, model_class_name)
    model = loader.get_model(asset, model_class_name)
    prediction = model.predict([MarketData(
        low_price='25810.4950896881', high_price='25921.976226367', close_price='25895.6782209584',
        timestamp=int(datetime.now().timestamp()), volume='5481314132.31'
    )])
    assert len(prediction) > 0, "Prediction is empty."
    assert isinstance(prediction[0], numpy.int64), f"Expected numpy.int64, got {type(prediction[0])}"
    assert prediction[0] in {0, 1}, f"Expected 0 or 1, got {prediction[0]}"


def test_model_training():
    with tempfile.TemporaryDirectory() as temp_model_dir:
        test_dataset_directory = './tests/datasets'
        pre_processor = CoinMarketCapPreProcessor()
        data_provider = LocalStorageDataProvider(directory=test_dataset_directory)
        asset = AssetEntity(
            id=2781, name='BTC', ticker_symbol='BTC',
            exchange='CRYPTO_DOT_COM', market_cap='123',
            decimal_places=8, keywords=[]
        )
        trainer = RandomForestClassifierTrainer(temp_model_dir, data_provider, pre_processor)
        trainer.train_and_save(asset)

        expected_model_path = os.path.join(temp_model_dir, "btc-random-forest.joblib")
        assert os.path.exists(expected_model_path), f"Model file not found at {expected_model_path}"
        assert os.path.getsize(expected_model_path) > 0, "Model file is empty"


@pytest.mark.skip("Skipping test_model_training_create_model as it is for creating sample model.")
def test_model_training_create_model():
    test_dataset_directory = './storage/datasets/'
    pre_processor = CoinMarketCapPreProcessor()
    data_provider = LocalStorageDataProvider(directory=test_dataset_directory)
    asset = AssetEntity(
        id=2781, name='BTC', ticker_symbol='BTC',
        exchange='CRYPTO_DOT_COM', market_cap='123',
        decimal_places=8, keywords=[]
    )
    trainer = RandomForestClassifierTrainer('./tests/models', data_provider, pre_processor)
    trainer.train_and_save(asset)
