import os
from unittest import TestCase

from sagemaker_inference import input_fn


class TestSitSagemakerInference(TestCase):
    def test_input_fn(self):
        # Arrange
        img_name = os.path.join(os.path.dirname(__file__), "imagesLFW", "George_W_Bush_0517.jpg")
        with open(img_name, "rb") as f:
            image_bytes = f.read()

        # Act
        actual = input_fn(image_bytes, "application/binary")

        # Assert
        self.assertSequenceEqual(image_bytes, actual)
