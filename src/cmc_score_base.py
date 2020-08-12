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


class CMCScoreBase:
    """
    Definition: https://www.nist.gov/sites/default/files/documents/2016/12/06/12_ross_cmc-roc_ibpc2016.pdf
    - Each sample is compared against all gallery samples. The resulting scores are sorted and ranked
    - Determine the rank at which a true match occurs
    - True Positive Identification Rate (TPIR): Probability of observing the correct identity within the top K ranks
    - CMC Curve: Plots TPIR against ranks
    - CMC Curve: Rank based metric
    """

    def score(self, pairwise_distance_matrix, target_label_y_query, target_label_x_gallery=None, k_threshold=5):
        raise NotImplementedError

    def get_top_k(self, pairwise_distance_matrix, k_threshold=5):
        raise NotImplementedError
