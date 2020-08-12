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

import logging
import os
import random

from datasets.triplet_datasetbase import TripletDatasetBase


class Market1501TripletDataset(TripletDatasetBase):
    """
    Returns 3 stream triplets
    """

    @property
    def num_classes(self):
        return len(self._zero_indexed_labels)

    def __init__(self, raw_directory, min_img_size_h=256, min_img_size_w=128, preprocessor=None):
        self.preprocessor = preprocessor
        self.min_img_size_w = min_img_size_w
        self.min_img_size_h = min_img_size_h
        self.raw_directory = raw_directory
        self.original_width = 64
        self.original_height = 128

        self._len = None
        self._files = [os.path.join(self.raw_directory, f) for f in os.listdir(self.raw_directory) if f.endswith("jpg")]

        # The market 1501 dataset files have the naming convention target_camerasite_..., e.g. 1038_c2s2_131202_03.jpeg
        self._target_raw_labels = [os.path.basename(f).split("_")[0] for f in self._files]
        self._zero_indexed_labels = {}
        self._label_indices_map = {}
        for i, rc in enumerate(self._target_raw_labels):
            self._zero_indexed_labels[rc] = self._zero_indexed_labels.get(rc, len(self._zero_indexed_labels))
            c = self._zero_indexed_labels[rc]
            # Create a label class to index map so it is easier for select item corresponding to label
            if c not in self._label_indices_map:
                self._label_indices_map[c] = []

            self._label_indices_map[c].append(i)

        self._classes = set(range(0, self.num_classes))

    def __len__(self):
        if self._len is None:
            self._len = len(self._target_raw_labels)

        return self._len

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __getitem__(self, index):
        """
        Returns triplets
        :param i:
        :return:
        """

        target = self._zero_indexed_labels[self._target_raw_labels[index]]

        # Randomly select an item that matches target class
        pos_index = random.choice(self._label_indices_map[target])
        # Negative class
        neg_class = random.choice(list(self._classes - {target}))
        neg_index = random.choice(self._label_indices_map[neg_class])

        pos_item = self._files[pos_index]
        q_item = self._files[index]
        neg_item = self._files[neg_index]

        self.logger.debug("preprocessing image {}".format(q_item))

        if self.preprocessor:
            pos_item = self.preprocessor(pos_item)
            q_item = self.preprocessor(q_item)
            neg_item = self.preprocessor(neg_item)

        return pos_item, q_item, neg_item, target
