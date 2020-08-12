import os
import tempfile
from unittest import TestCase

import torch

from predictor import Predictor
from train_factory import TrainFactory
from triplet_dataset_factory_service_locator import TripletDatasetFactoryServiceLocator


class TestSitPredictor(TestCase):

    def test___call__(self):
        # Arrange
        output_dir = tempfile.mkdtemp()
        self._run_train(output_dir)
        sut = Predictor(output_dir)
        img_name = os.path.join(os.path.dirname(__file__), "imagesMarket1501", "0007_c2s3_070952_01.jpg")

        # Act
        result = sut(img_name)

        # Assert
        self.assertIsInstance(result, torch.Tensor)

    def _run_train(self, output_dir):
        img_dir = os.path.join(os.path.dirname(__file__), "imagesMarket1501")
        dataset_factory = TripletDatasetFactoryServiceLocator().get_factory("Market1501TripletFactory")
        dataset = dataset_factory.get(img_dir)

        factory = TrainFactory(num_workers=1, epochs=2, batch_size=6, early_stopping=True, patience_epochs=2)
        pipeline = factory.get(dataset)
        pipeline.run(dataset, dataset, output_dir)
