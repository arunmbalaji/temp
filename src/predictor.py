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
import glob
import os
import tempfile

import torch
from PIL import Image
from torch import load
from torchvision.transforms import transforms


class Predictor:
    """
    Runs predictions on a model
    """

    def __init__(self, model_dir_or_filepath, min_img_size_h=214, min_img_size_w=214):
        self.min_img_size_w = min_img_size_w
        self.min_img_size_h = min_img_size_h
        model_file = model_dir_or_filepath
        if os.path.isdir(model_dir_or_filepath):
            model_dir_or_filepath = model_dir_or_filepath.rstrip("/")
            model_file = self._find_artifact(model_dir_or_filepath)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = load(model_file, map_location=self.device)
        self.model.to(device=self.device)

    def __call__(self, input_file_or_bytes):
        """
Runs predictions on input data, which can be a file path or an array of bytes  or tensor
        :param input_file_or_bytes: A file path or an array of bytes or tensor
        :return:
        """
        assert isinstance(input_file_or_bytes, str) or isinstance(input_file_or_bytes, bytes) or isinstance(
            input_file_or_bytes,
            torch.Tensor), "Expected the type of arg input_file_or_bytes to be either str of bytes or Tensor, but found {}".format(
            type(input_file_or_bytes))

        # If file
        if isinstance(input_file_or_bytes, str):
            input_data = self._pre_process_image(input_file_or_bytes)
        # Else bytes
        elif isinstance(input_file_or_bytes, bytes):
            with tempfile.NamedTemporaryFile("w+b") as f:
                f.write(input_file_or_bytes)
                f.seek(0)
                input_data = self._pre_process_image(f)
        else:
            input_data = input_file_or_bytes

        # run inference
        self.model.eval()
        with torch.no_grad():
            return self.model(input_data.to(self.device))

    @staticmethod
    def _find_artifact(model_dir):
        pattern = "{}/*.pt".format(model_dir)
        matching = glob.glob(pattern)

        assert len(
            matching) == 1, "Expected exactly one in file that ends with either .pt, but found {}".format(
            pattern,
            len(matching))
        matched_file = matching[0]
        return matched_file

    # TODO: This is repeat code block as seen within dataset class, refactor
    def _pre_process_image(self, image):
        # pre-process data
        image = Image.open(image)
        # The min size, as noted in the PyTorch pretrained models doc, is 224 px.
        transform_pipeline = transforms.Compose([transforms.Resize((self.min_img_size_h, self.min_img_size_w)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                      # torch image: C X H X W
                                                                      std=[0.229, 0.224, 0.225])])
        img_tensor = transform_pipeline(image)
        # Add batch [N, C, H, W]
        img_tensor = img_tensor.unsqueeze(0)
        return img_tensor
