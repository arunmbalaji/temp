from unittest import TestCase

import torch

from euclidean_pairwise_distance import EuclideanPairwiseDistance


class TestEuclideanPairwiseDistance(TestCase):

    def test___call__x(self):
        """
        Test case where x is the same as y
        :return:
        """
        # Arrange
        x = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]], dtype=torch.float32)
        sut = EuclideanPairwiseDistance()
        expected = torch.tensor([[0.0, 3.0, 12.0, 27.0],
                                 [3.0, 0.0, 3.0, 12.0],
                                 [12.0, 3.0, 0.0, 3.0],
                                 [27.0, 12, 3, 0.0]])

        # Act
        actual = sut(x)

        # Assert
        self.assertTrue(torch.equal(actual, expected))

    def test___call__x_y(self):
        """
        pass y separately
        :return:
        """
        """
        :return: 
        """
        # Arrange
        x = torch.tensor([[1, 1, 1]], dtype=torch.float32)
        y = torch.tensor([[2, 2, 2], [3, 3, 3]], dtype=torch.float32)
        sut = EuclideanPairwiseDistance()
        expected = torch.tensor([[3.0, 12.0]])

        # Act
        actual = sut(x, y)

        # Assert
        self.assertTrue(torch.equal(actual, expected))

    def test___call__x_y_smaller(self):
        """
        pass y separately, by  is smaller
        :return:
        """
        """
        :return: 
        """
        # Arrange
        x = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
        y = torch.tensor([[3, 3, 3]], dtype=torch.float32)
        sut = EuclideanPairwiseDistance()
        expected = torch.tensor([[12.0], [3.0]])

        # Act
        actual = sut(x, y)

        # Assert
        self.assertTrue(torch.equal(actual, expected))
