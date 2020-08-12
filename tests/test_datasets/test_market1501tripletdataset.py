import os
from unittest import TestCase

from datasets.market1501_triplet_dataset import Market1501TripletDataset


class TestMarket1501TripletDataset(TestCase):

    def test___len__(self):
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "..", "imagesMarket1501")
        sut = Market1501TripletDataset(img_dir)
        expected = 6

        # Act
        actual = len(sut)

        # Assert
        self.assertEqual(expected, actual)

    def test___call___neg_item(self):
        """
        Case to make sure negative item is class does not match positive item
        :return:
        """
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "..", "imagesMarket1501")
        sut = Market1501TripletDataset(img_dir)

        f = lambda f: os.path.basename(f)[0:os.path.basename(f).index("_")]

        # Act
        p, q, n, target = sut[1]

        # Assert
        self.assertNotEqual(f(q), f(n))
        self.assertNotEqual(f(p), f(n))

    def test___call___pos_item(self):
        """
        Case to make sure class of q and p match
        :return:
        """
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "..", "imagesMarket1501")
        sut = Market1501TripletDataset(img_dir)

        f = lambda f: os.path.basename(f)[0:os.path.basename(f).index("_")]

        # Act
        p, q, n, target = sut[1]

        # Assert class of positive and q match
        self.assertEqual(f(q), f(p))

    def test___call___target(self):
        """
        Case to make sure class target q is returned
        :return:
        """
        # Arrange
        img_dir = os.path.join(os.path.dirname(__file__), "..", "imagesMarket1501")
        sut = Market1501TripletDataset(img_dir)

        f = lambda f: os.path.basename(f)[0:os.path.basename(f).index("_")]

        # Act
        p, q, n, target = sut[1]

        # Assert class of positive and q match
        self.assertEqual(sut._zero_indexed_labels[f(q)], target)
