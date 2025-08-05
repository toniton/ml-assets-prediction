import os
import tempfile
from datetime import datetime

import numpy

from api import RandomForestClassifierModel
from api.interfaces.market_data import MarketData
from src.entities.asset_entity import AssetEntity
from src.providers.clients.local_storage_data_provider import LocalStorageDataProvider
from src.providers.preprocessors.coinmarketcap_preprocessor import CoinMarketCapPreProcessor
from src.training.models.random_forest_classifier_trainer import RandomForestClassifierTrainer


def test_model_prediction():
    asset = AssetEntity(name='BTC', ticker_symbol='BTC', market_cap='123', decimal_places=8, keywords=[])
    classifier = RandomForestClassifierModel(asset, './models').load_model()
    prediction = classifier.predict(MarketData(
        low_price='1', high_price='2', close_price='1.5',
        timestamp=int(datetime.now().timestamp()), volume='10000'
    ))
    assert len(prediction) > 0, "Prediction is empty."
    assert isinstance(prediction[0], numpy.int64), f"Expected numpy.int64, got {type(prediction[0])}"
    assert prediction[0] in (0, 1), f"Expected 0 or 1, got {prediction[0]}"


def test_model_training():
    with tempfile.TemporaryDirectory() as temp_model_dir:
        test_dataset_directory = './datasets'
        pre_processor = CoinMarketCapPreProcessor()
        data_provider = LocalStorageDataProvider(directory=test_dataset_directory)
        asset = AssetEntity(name='BTC', ticker_symbol='BTC', market_cap='123', decimal_places=8, keywords=[])
        trainer = RandomForestClassifierTrainer(temp_model_dir, data_provider, pre_processor)
        trainer.train_and_save(asset)

        expected_model_path = os.path.join(temp_model_dir, "btc-random-forest.joblib")
        assert os.path.exists(expected_model_path), f"Model file not found at {expected_model_path}"
        assert os.path.getsize(expected_model_path) > 0, "Model file is empty"
