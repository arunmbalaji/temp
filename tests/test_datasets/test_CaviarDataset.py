import os
from unittest import TestCase

from datasets.caviar_dataset import CaviarDataset


class TestCaviarDataset(TestCase):

    def test___len__(self):
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "..", "imagesCaviar")
        sut = CaviarDataset(img_dir)
        expected = 8

        # Act
        actual = len(sut)

        # Assert
        self.assertEqual(expected, actual)

    def test___call__(self):
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "..", "imagesCaviar")
        sut = CaviarDataset(img_dir)
        total_images = 8
        expected_classes = 2

        # Act
        actual_items = [sut[i][1] for i in range(total_images)]

        # Assert
        self.assertEqual(expected_classes - 1, max(actual_items),
                         "The max target class index must be zero-indexed and 1 less than the max number of classes")
