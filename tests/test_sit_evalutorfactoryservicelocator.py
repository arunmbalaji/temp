# *************************************************************
# * Copyright 2019 Amazon.com, Inc. and its affiliates. All Rights Reserved.
# 
# Licensed under the Amazon Software License (the "License").
#  You may not use this file except in compliance with the License.
# A copy of the License is located at
# 
#  http://aws.amazon.com/asl/
# 
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
# ***************************************************************
from unittest import TestCase

from datasets.evaluation_dataset_factorybase import EvaluationDatasetFactoryBase
from evaluator_factory_service_locator import EvalutorFactoryServiceLocator
from evaluators.evaluator_factory_base import EvaluatorFactoryBase


class TestSitEvaluatorFactoryServiceLocator(TestCase):

    def test_factory_names(self):
        # Arrange
        sut = EvalutorFactoryServiceLocator()

        # act
        class_names = sut.factory_names

        # assert
        self.assertEqual(len(class_names), 1,
                         " The number of expected dataset factory classes doesnt match.. Check the number of classes that inhert from {}  !".format(
                             type(EvaluationDatasetFactoryBase)))

    def test_get_factory(self):
        # Arrange
        sut = EvalutorFactoryServiceLocator()
        class_names = sut.factory_names

        # Act
        obj = sut.get_factory(class_names[0])

        # assert
        self.assertIsInstance(obj, EvaluatorFactoryBase)
