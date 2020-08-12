from unittest import TestCase

import torch

from cmc_score import CMCScore


class TestCMCScore(TestCase):
    def test_score(self):
        # Arrange
        pairwise_distance = torch.tensor([[0, 11112, 1], [11112, 0, 3], [1, 3, 0]])
        sut = CMCScore()
        expected = 66.67

        # Act
        actual = sut.score(pairwise_distance, torch.tensor([2, 1, 2]), k_threshold=2)

        # Assert
        self.assertEqual(round(actual, 2), expected)

    def test_score_x_y_smaller_gallery(self):
        """Case where x (gallaer) is smaller then query"""
        # Arrange
        pairwise_distance = torch.tensor([[0, 11112, 1], [11112, 0, 3], [1, 3, 0], [1, 999, 0]])
        target_y_query = torch.tensor([2, 1, 2, 1])
        target_x_gallery = torch.tensor([1, 1, 2])

        sut = CMCScore()
        expected = 100.0 * 2 / 4

        # Act
        actual = sut.score(pairwise_distance, target_y_query, target_x_gallery, k_threshold=1)

        # Assert
        self.assertEqual(round(actual, 2), expected)

    def test_score_x_y_larger_gallery(self):
        """Case where x (gallery) is larger then query"""
        # Arrange
        pairwise_distance = torch.tensor([[0, 11112, 1], [11112, 0, 3]])
        target_y_query = torch.tensor([2, 1])
        target_x_gallery = torch.tensor([1, 1, 2])

        sut = CMCScore()
        expected = 100.0 * 1 / 2

        # Act
        actual = sut.score(pairwise_distance, target_y_query, target_x_gallery, k_threshold=1)

        # Assert
        self.assertEqual(round(actual, 2), expected)
