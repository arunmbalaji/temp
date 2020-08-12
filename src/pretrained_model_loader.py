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
import argparse
import inspect
import os

import torch
import torchvision.models


class PretrainedModelLoader:
    """
    Loads a pretrained model
    """

    def __init__(self):
        self._valid_models = [n for n, _ in inspect.getmembers(torchvision.models, inspect.isfunction)]

    @property
    def model_names(self):
        return self._valid_models

    def load(self, model_name):
        """
Returns a pretrained model based on the model_name argument. The model name should be one of the models defined in torchvision.models.
        :param model_name: The model name defined in torchvision.models. E.g. 'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'inception_v3', 'resnet101', 'resnet152', 'resnet18', 'resnet34', 'resnet50', 'squeezenet1_0', 'squeezenet1_1' etc
        :return: Pytorch model object
        """
        # Check if the model name passed is a callable
        model_func = getattr(torchvision.models, model_name, None)
        assert callable(
            model_func) == True, "The function torchvision.models.{} must be a callable. The valid list of callables are {}".format(
            model_name, self._valid_models)

        # Create model
        model = model_func(pretrained=True)

        return model

    def save(self, model, model_dir):
        model_name_path = os.path.join(model_dir, "model.pt")
        torch.save(model, model_name_path)
        return model_name_path

    def __call__(self, model_name, model_dir):
        return self.save(self.load(model_name), model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--modelname",
                        help="The name of the petrained model as defined in torchvision.models",
                        choices=PretrainedModelLoader().model_names, required=True)

    parser.add_argument("--modelpath",
                        help="The model path", required=True)

    args = parser.parse_args()

    print(args.__dict__)

    model_name_path = PretrainedModelLoader()(args.modelname, args.modelpath)

    print("Model save completed and can be found in {} ".format(model_name_path))
