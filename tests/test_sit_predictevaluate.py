# *****************************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.  *
#                                                                             *
# Licensed under the Amazon Software License (the "License").                 *
#  You may not use this file except in compliance with the License.           *
# A copy of the License is located at                                         *
#                                                                             *
#  http://aws.amazon.com/asl/                                                 *
#                                                                             *
#  or in the "license" file accompanying this file. This file is distributed  *
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either  *
#  express or implied. See the License for the specific language governing    *
#  permissions and limitations under the License.                             *
# *****************************************************************************

import os
import tempfile
from unittest import TestCase

from main_predict_evaluate import PredictEvaluate
from train_factory import TrainFactory
from triplet_dataset_factory_service_locator import TripletDatasetFactoryServiceLocator


class TestSitPredictEvaluate(TestCase):

    def test___call__(self):
        # Arrange
        output_dir = tempfile.mkdtemp()
        images_dir = os.path.join(os.path.dirname(__file__), "imagesMarket1501")
        eval_dataset_factory = "Market1501Factory"
        train_dataset_factory = "Market1501TripletFactory"

        self._run_train(images_dir, train_dataset_factory, output_dir)
        sut = PredictEvaluate()

        # Act
        result = sut(eval_dataset_factory, model_path=output_dir, query_images_dir=images_dir)

        # Assert
        self.assertIsInstance(result, float)

    def test___call__query(self):
        # Arrange
        output_dir = tempfile.mkdtemp()
        train_images_dir = os.path.join(os.path.dirname(__file__), "imagesMarket1501")
        query_images_dir = os.path.join(os.path.dirname(__file__), "imagesMarket1501", "query")
        gallery_images_dir = train_images_dir

        eval_dataset_factory = "Market1501Factory"
        train_dataset_factory = "Market1501TripletFactory"

        self._run_train(train_images_dir, train_dataset_factory, output_dir)
        sut = PredictEvaluate()

        # Act
        result = sut(eval_dataset_factory, model_path=output_dir, gallery_images_dir=gallery_images_dir,
                     query_images_dir=query_images_dir)

        # Assert
        self.assertIsInstance(result, float)

    def _run_train(self, images_dir, dataset_factory, output_dir):
        dataset_factory = TripletDatasetFactoryServiceLocator().get_factory(dataset_factory)
        dataset = dataset_factory.get(images_dir)

        factory = TrainFactory(num_workers=1, epochs=2, batch_size=6, early_stopping=True, patience_epochs=2)
        pipeline = factory.get(dataset)
        pipeline.run(dataset, dataset, output_dir)
