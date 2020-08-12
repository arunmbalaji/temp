import os
from unittest import TestCase

from datasets.market1501_dataset import Market1501Dataset


class TestMarket1501Dataset(TestCase):

    def test___len__(self):
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "..", "imagesMarket1501")
        sut = Market1501Dataset(img_dir)
        expected = 6

        # Act
        actual = len(sut)

        # Assert
        self.assertEqual(expected, actual)

    def test___call__(self):
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "..", "imagesMarket1501")
        sut = Market1501Dataset(img_dir)
        total_images = 6
        expected_classes = 2

        # Act
        actual_items = [sut[i][1] for i in range(total_images)]

        # Assert
        self.assertEqual(expected_classes - 1, max(actual_items),
                         "The max target class index must be zero-indexed and 1 less than the max number of classes")
