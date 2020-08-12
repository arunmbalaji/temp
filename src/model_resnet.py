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
from torch import nn
from torchvision import models


class ModelResnet(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet_model = models.resnet50(pretrained=True)
        # Change the final layer so that the number of classes
        # Use print final layer to figure out the input size to the final layer
        # print(self.model.fc)
        # fc_input_size = 512  # 2048 is for resnet 50
        # self.resnet_model.fc = nn.Sequential(nn.Linear(fc_input_size, self.embed_dim))

    def forward(self, input):
        fc_out = self.resnet_model(input)
        return fc_out
