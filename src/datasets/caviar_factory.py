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

from datasets.caviar_dataset import CaviarDataset
from datasets.evaluation_dataset_factorybase import EvaluationDatasetFactoryBase


class CaviarFactory(EvaluationDatasetFactoryBase):

    def get(self, query_images, gallery_images=None):
        gallery_images = gallery_images or query_images
        query_dataset = CaviarDataset(query_images, min_img_size_h=256, min_img_size_w=128)
        gallery_dataset = CaviarDataset(gallery_images, min_img_size_h=256, min_img_size_w=128,
                                        initial_label_map=query_dataset.label_number_map)

        return query_dataset, gallery_dataset
