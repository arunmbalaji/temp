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
import logging
import sys

import torch
from torch.utils.data import DataLoader

from evaluation_dataset_factory_service_locator import EvaluationDatasetFactoryServiceLocator
from evaluator_factory_service_locator import EvalutorFactoryServiceLocator
from predictor import Predictor


class PredictEvaluate:

    def __call__(self, dataset_factory_name, model_path, query_images_dir, gallery_images_dir=None,
                 eval_factory_name="EvaluationFactory"):
        # Construct factories
        evalfactory = EvalutorFactoryServiceLocator().get_factory(eval_factory_name)
        evaluator = evalfactory.get_evaluator()
        datasetfactory = EvaluationDatasetFactoryServiceLocator().get_factory(dataset_factory_name)

        query_dataset, gallery_dataset = datasetfactory.get(query_images_dir, gallery_images_dir)

        # get query embeddings
        class_person_query, embeddings_query = self._get_predictions(query_dataset, model_path)

        # Get gallery embeddings
        class_person_gallery, embeddings_gallery = self._get_predictions(gallery_dataset, model_path)

        # Evaluate
        result = evaluator(query_embedding=embeddings_query,
                           query_target_class=class_person_query,
                           gallery_embedding=embeddings_gallery,
                           gallery_target_class=class_person_gallery)
        return result

    def _get_predictions(self, dataset, model_path):
        """
        Returns predictions for the dataset
        :param dataset: Dataset
        :param model_path: Model path to use for predictions
        :return:
        """
        batch_size = min(len(dataset), 32)
        dataloader_query = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        model = Predictor(model_path)
        embeddings = []
        class_person = []
        for person_img, target in dataloader_query:
            embedding = model(person_img)
            embeddings.extend(embedding)
            class_person.extend(target)

        embeddings = torch.stack(embeddings)
        class_person = torch.stack(class_person)
        return class_person, embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",
                        help="The type of dataset",
                        choices=EvaluationDatasetFactoryServiceLocator().factory_names, required=True)

    parser.add_argument("--modelpath",
                        help="The model path", required=True)

    parser.add_argument("--queryimagesdir",
                        help="The directory path containing query dir", required=True)

    parser.add_argument("--galleryimagesdir",
                        help="The directory path containing gallery dataset", default=None)

    parser.add_argument("--log-level", help="Log level", default="INFO", choices={"INFO", "WARN", "DEBUG", "ERROR"})

    args = parser.parse_args()

    print(args.__dict__)

    # Set up logging
    logging.basicConfig(level=logging.getLevelName(args.log_level), handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    result = PredictEvaluate()(args.dataset, args.modelpath, args.queryimagesdir, args.galleryimagesdir)
    print("Score is {}".format(result))
