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

from __future__ import absolute_import

import torch


class EuclideanPairwiseDistance():
    """
    Computes pairwise euclidean distance
    """

    def __call__(self, x, y=None):
        """
Computes pairwise euclidean distance
        :param x: n x f float matrix  ( n samples and f features)
        :param y: optional y matrix
        :return: pair wise euclidean distance
        """
        assert len(x.shape) == 2, "Requires a 2D matrix"

        # Note: not using (x-x.unsqueeze(1))^2 as it creates a very large matrix

        squared_x = torch.pow(x, 2).sum(1)

        if y is not None:
            yt = torch.t(y)
            squared_y = torch.pow(y, 2).sum(1)
        else:
            yt = torch.t(x)
            squared_y = squared_x

        xy = x @ yt
        sum_of_squares = squared_x.unsqueeze(1) + squared_y - 2 * xy
        result = sum_of_squares  # torch.sqrt(sum_of_squares)

        return result
