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

from cmc_score import CMCScore
from euclidean_pairwise_distance import EuclideanPairwiseDistance
from evaluators.evaluator import Evaluator
from evaluators.evaluator_factory_base import EvaluatorFactoryBase


class EvaluationFactory(EvaluatorFactoryBase):
    """
    Creates a evaluator
    """

    def __init__(self, k_threshold=1):
        self.k_threshold = k_threshold

    def get_evaluator(self):
        distance_metric = EuclideanPairwiseDistance()
        scorer = CMCScore()

        evaluator = Evaluator(distance_metric, scorer, k_threshold=self.k_threshold)

        return evaluator
