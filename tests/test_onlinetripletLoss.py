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

from online_tripletloss import OnlineTripletLoss


class TestOnlineTripletLoss(TestCase):
    def test_forward_max_nonzero(self):
        """
        Case where the difference between p and n sample is much smaller than the margin
        :return:
        """
        # Arrange
        margin = 50
        topk = 25
        sut = OnlineTripletLoss(margin, topk)
        predicted = torch.tensor([[0, 0, 1], [0, 0, 1], [1, 1, 2]], dtype=torch.float)

        target = torch.tensor([1, 1, 0])

        expected = 0 - 3 + margin

        # Act
        actual = sut.forward(predicted, target)

        # Assert
        self.assertEqual(round(expected, 2), round(actual.item(), 2))

    def test_forward_max_zero(self):
        """
        Case where the difference between p and n sample is much greater than the margin
        :return:
        """
        # Arrange
        margin = .1
        sut = OnlineTripletLoss(margin)
        predicted = torch.tensor([[0, 0, 1], [0, 0, 1], [1, 1, 2]], dtype=torch.float)

        target = torch.tensor([1, 1, 0])

        # max  ( 0, 0-3 + margin)
        expected = 0

        # Act
        actual = sut.forward(predicted, target)

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
        sut = OnlineTripletLoss(.5)

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
        sut = OnlineTripletLoss(.5)

        # Act
        actual = sut._get_distance(input_x, input_y)

        # Assert
        self.assertSequenceEqual(expected.cpu().numpy().round(2).tolist(), actual.cpu().numpy().round(2).tolist())

    def test__generate_all_triplets_single(self):
        """
        Case where just one positive  example and 1 negative sample
        """
        # Arrange
        input_target = torch.tensor([0, 0, 1])
        expected_triplet_indices = torch.tensor([[0, 1, 2]])
        sut = OnlineTripletLoss(.5)

        # Act
        actual = sut._generate_all_triplets(input_target)

        # Assert
        self.assertSequenceEqual(expected_triplet_indices.shape, actual.shape)
        self.assertSequenceEqual(expected_triplet_indices.cpu().numpy().tolist(), actual.cpu().numpy().tolist())

    def test__generate_all_triplets_single_reverse(self):
        """
        Case where just one positive  example and 1 negative sample
        """
        # Arrange
        input_target = torch.tensor([1, 0, 0])
        expected_triplet_indices = torch.tensor([[1, 2, 0]])
        sut = OnlineTripletLoss(.5)

        # Act
        actual = sut._generate_all_triplets(input_target)

        # Assert
        self.assertSequenceEqual(expected_triplet_indices.cpu().numpy().tolist(), actual.cpu().numpy().tolist())

    def test__generate_all_triplets_three_clases(self):
        """
        Case where just 2 positive  example and 1 negative sample
        """
        # Arrange
        input_target = torch.tensor([0, 0, 1, 2])
        expected_triplet_indices = torch.tensor([[0, 1, 2], [0, 1, 3]])
        sut = OnlineTripletLoss(.5)

        # Act
        actual = sut._generate_all_triplets(input_target)

        # Assert
        self.assertSequenceEqual(expected_triplet_indices.cpu().numpy().tolist(), actual.cpu().numpy().tolist())

    def test__generate_all_triplets_three_clases_missing_target(self):
        """
        Case where not all target classes are presented in the target ( can happen within a batch)
        """
        # Arrange
        input_target = torch.tensor([0, 0, 1, 4])
        expected_triplet_indices = torch.tensor([[0, 1, 2], [0, 1, 3]])
        sut = OnlineTripletLoss(.5)

        # Act
        actual = sut._generate_all_triplets(input_target)

        # Assert
        self.assertSequenceEqual(expected_triplet_indices.cpu().numpy().tolist(), actual.cpu().numpy().tolist())

    def test__generate_all_triplets_multiple_negative(self):
        """
        Case where just one positive  example and 1 negative sample
        """
        # Arrange
        input_target = torch.tensor([0, 0, 1, 2])
        expected_triplet_indices = torch.tensor([[0, 1, 2], [0, 1, 3]])
        sut = OnlineTripletLoss(.5)

        # Act
        actual = sut._generate_all_triplets(input_target)

        # Assert
        self.assertSequenceEqual(expected_triplet_indices.cpu().numpy().tolist(), actual.cpu().numpy().tolist())
