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

import torch

from tripletloss import TripletLoss


class TestTripletLoss(TestCase):
    def test_forward_max_nonzero(self):
        """
        Case where the difference between p and n sample is much smaller than the margin
        :return:
        """
        # Arrange
        margin = 50
        topk = 25
        sut = TripletLoss(margin, topk)
        p = torch.tensor([[0, 0, 1]], dtype=torch.float)
        q = torch.tensor([[0, 0, 1]], dtype=torch.float)
        n = torch.tensor([[1, 1, 2]], dtype=torch.float)

        target = torch.tensor([1])

        expected = 0 - 3 + margin

        # Act
        actual = sut.forward(p, q, n, target)

        # Assert
        self.assertEqual(round(expected, 2), round(actual.item(), 2))

    def test_forward_max_zero(self):
        """
        Case where the difference between p and n sample is much greater than the margin
        :return:
        """
        # Arrange
        margin = .1
        sut = TripletLoss(margin)
        p = torch.tensor([[0, 0, 1]], dtype=torch.float)
        q = torch.tensor([[0, 0, 1]], dtype=torch.float)
        n = torch.tensor([[1, 1, 2]], dtype=torch.float)

        target = torch.tensor([1])

        # max  ( 0, 0-3 + margin)
        expected = 0

        # Act
        actual = sut.forward(p, q, n, target)

        # Assert
        self.assertEqual(round(expected, 2), round(actual.item(), 2))

    def test__get_distance_zero(self):
        """
        Case where just a single item in each array that are the same
        """
        # Arrange
        input_x = torch.tensor([[0, 0, 1]], dtype=torch.float)
        input_y = torch.tensor([[0, 0, 1]], dtype=torch.float)

        expected = torch.tensor([0])
        sut = TripletLoss(.5)

        # Act
        actual = sut._get_distance(input_x, input_y)

        # Assert
        self.assertSequenceEqual(expected.cpu().numpy().tolist(), actual.cpu().numpy().tolist())

    def test__get_distance_single(self):
        """
        Case where just a single item in each array
        """
        # Arrange
        input_x = torch.tensor([[1, 8, 7]], dtype=torch.float)
        input_y = torch.tensor([[2, 3, 4]], dtype=torch.float)

        expected = torch.tensor([35])
        sut = TripletLoss(.5)

        # Act
        actual = sut._get_distance(input_x, input_y)

        # Assert
        self.assertSequenceEqual(expected.cpu().numpy().round(2).tolist(), actual.cpu().numpy().round(2).tolist())
