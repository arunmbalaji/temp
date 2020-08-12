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

import torch
from torch import nn


class TripletLoss(nn.Module):
    """
    Implement  triplet loss based on hard triplet mining strategy in the paper - End-to-end Learning of Deep Visual Representations for Image Retrieval  - https://arxiv.org/pdf/1610.07940.pdf


    ``To ensure that the sampled triplets are useful, we first select randomly N training samples, extract their features with the current model, and compute all possible triplets and their losses, which is fast once the features have been extracted.
    All the triplets that incur a loss are prese- lected as good candidates. Triplets can then be sampled from that set of good candidates, with a bias towards hard triplets, i.e. triplets that produce a high loss.
    In practice this is achieved by randomly sampling one of the N images with a uniform distribution and then ran- domly choosing one of the 25 triplets with the largest loss that involve that particular image as a query.
    Note that, in theory, one should recompute the set of good candidates every time the model gets updated, which is very time consuming.
    In practice, we assume that most of the hard triplets for a given model will remain hard even if the model gets updated a few times, and there- fore we only update the set of good candidates after the model has been updated k times.
    We used N = 5000 samples and k = 64 iterations with a batch size of 64 triplets per iteration in our experiments.
    ``
    The loss function for query sample q, postive sample p  and negative sample n  is and N total samples
    .. math::
        L (q, p, q) =  1/N \Sigma max(d(q, p) - d(q, n) + margin, 0)

    Useful resources
    ================

    * https://omoindrot.github.io/triplet-loss
    """

    def __init__(self, margin, topk=None):
        super().__init__()
        self.topk = topk
        self.margin = margin

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def forward(self, pos_embedding, query_embedding, neg_embedding, target):
        """
Computes the triplet loss
        :param neg_embedding: a 2D tensor of embeddings negtive sample matching that does not match query class
        :param pos_embedding: a 2D tensor of embeddings positive sample matching query class
        :param query_embedding: a 2D tensor of embeddings
        :param target: 1d tensor of target
        :return: loss
        """

        pos_distance = self._get_distance(pos_embedding, query_embedding)
        neg_distance = self._get_distance(neg_embedding, query_embedding)

        # use relu instead of max
        losses = torch.relu(pos_distance - neg_distance + self.margin)

        # Filter hard loss
        if self.topk is not None:
            losses = torch.topk(losses, k=min(losses.shape[0], self.topk))[0]

        loss = losses.mean()
        return loss

    def _get_distance(self, x, y):
        """
        Returns the euclidean distance between x  and y where x and y are 2D tensors and the length of xand y is the same
        :param x: 2d tensor
        :param y: 2d tensor
        :return: distance along the 1 dim
        """
        assert x.shape == y.shape, "Expecting the shapes of x and y to match"

        result = torch.pow(x - y, 2).sum(1)

        return result
