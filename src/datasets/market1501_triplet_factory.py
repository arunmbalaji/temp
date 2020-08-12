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

from datasets.market1501_triplet_dataset import Market1501TripletDataset
from datasets.triplet_dataset_factorybase import TripletDatasetFactoryBase
from image_preprocessor import ImagePreprocessor


class Market1501TripletFactory(TripletDatasetFactoryBase):

    def __init__(self):
        pass

    def get(self, images_dir):
        # Market150 dataset size is 64 width, height is 128, so we maintain the aspect ratio
        # NOTE: for some reason oly 224 / 224 works, any other shape results in NAN
        dataset = Market1501TripletDataset(images_dir, min_img_size_h=256, min_img_size_w=128)
        processor = ImagePreprocessor(min_img_size_h=dataset.min_img_size_h, min_img_size_w=dataset.min_img_size_w,
                                      original_height=dataset.original_height, original_width=dataset.original_width)

        dataset.preprocessor = processor
        return dataset
