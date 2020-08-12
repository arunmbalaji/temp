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

from evaluators.evaluator_base import EvaluatorBase


class Evaluator(EvaluatorBase):
    def __init__(self, distance_measurer, scorer, k_threshold=1):
        self.k_threshold = k_threshold
        self.distance_metric = distance_measurer
        self.scorer = scorer

    def __call__(self, query_embedding, query_target_class, gallery_embedding=None, gallery_target_class=None):
        # Compute pairwise
        pairwise_distance = self.distance_metric(query_embedding, gallery_embedding)

        score = self.scorer.score(pairwise_distance, target_label_y_query=query_target_class,
                                  target_label_x_gallery=gallery_target_class, k_threshold=self.k_threshold)
        return score
