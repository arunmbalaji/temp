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
import logging

import numpy as np
import torch

from cmc_score_base import CMCScoreBase


class CMCScore(CMCScoreBase):
    """
    Definition: https://www.nist.gov/sites/default/files/documents/2016/12/06/12_ross_cmc-roc_ibpc2016.pdf
    - Each sample is compared against all gallery samples. The resulting scores are sorted and ranked
    - Determine the rank at which a true match occurs
    - True Positive Identification Rate (TPIR): Probability of observing the correct identity within the top K ranks
    - CMC Curve: Plots TPIR against ranks
    - CMC Curve: Rank based metric
    """

    def get_top_k(self, pairwise_distance_matrix, k_threshold=5):
        # Set nan to zero...
        # pairwise_distance_matrix[torch.isnan(pairwise_distance_matrix)] = 0

        # Get the index matrix of the top k nearest neighbours
        rank_k = torch.topk(pairwise_distance_matrix, k=k_threshold, dim=1, largest=False)[1]

        return rank_k

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def accuracy_at_top_k(self, pairwise_distance_matrix, target_label_y_query, k, target_label_x_gallery=None):
        top_k = k

        # Ignore rank 0, as they are the same elements ( diagonal), so add 1 to k
        if target_label_x_gallery is None:
            top_k += 1
        rank_k = self.get_top_k(pairwise_distance_matrix, top_k)

        # Ignore rank 0, as they are the same elements ( diagonal)
        if target_label_x_gallery is None:
            rank_k = rank_k[:, 1:]
            target_label_x_gallery = target_label_y_query

        self.logger.info("Running query images {} against gallery of size {}".format(len(target_label_y_query),
                                                                                     len(target_label_x_gallery)))

        target_label_y_query = target_label_y_query.cpu().numpy()
        target_label_x_gallery = target_label_x_gallery.cpu().numpy()

        # map item index to target class
        map_id_to_class = lambda x: target_label_x_gallery[x]
        map_id_to_class_vec = np.vectorize(map_id_to_class)
        rank_k_label_x_gallery = map_id_to_class_vec(rank_k.cpu())

        # Compute accuracy
        correct = 0
        for i, r in enumerate(rank_k_label_x_gallery):
            if target_label_y_query[i] in r: correct += 1

        accuracy = (correct * 100.0) / len(target_label_y_query)

        return accuracy, rank_k, rank_k_label_x_gallery

    def score(self, pairwise_distance_matrix, target_label_y_query, target_label_x_gallery=None, k_threshold=5):

        """
    Computes CMC Score.
        :param target_label_y_query: The target classes of the query samples in y axis
        :param target_label_x_gallery: The target classes of the gallery samples in x axis
        :param pairwise_distance_matrix:Diagonal matrix with distance measure
        :param target_label: the target label ( labels must be zero indexed integers)
        :param k_threshold: max K to use for averaging
        :return:
        """
        assert pairwise_distance_matrix.shape[0] == target_label_y_query.shape[
            0], "The size of the target_y_query labels {} should match the length of pairwise_distance_matrix {}".format(
            target_label_y_query.shape[0], pairwise_distance_matrix.shape[0])

        if target_label_x_gallery is not None:
            assert pairwise_distance_matrix.shape[1] == target_label_x_gallery.shape[
                0], "The size of the target_x_gallery labels {} should match the length of pairwise_distance_matrix {}".format(
                target_label_x_gallery.shape[0], pairwise_distance_matrix.shape[1])

        total = 0.0
        for k in range(1, k_threshold + 1):
            accuracy, _, _ = self.accuracy_at_top_k(pairwise_distance_matrix, target_label_y_query,
                                                    k, target_label_x_gallery)
            total += accuracy

        return total / k_threshold
