import json
import os

import boto3
from locust import HttpLocust, TaskSet, task


class SageMakerConfig:

    def __init__(self):
        self.__config__ = None

    @property
    def endpointname(self):
        return self.config["endpointName"]

    @property
    def data(self):
        return self.config["dataPayload"]

    @property
    def config(self):
        self.__config__ = self.__config__ or self.load_config()
        return self.__config__

    def load_config(self):
        config_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
        with open(config_file, "r") as c:
            return json.loads(c.read())


class SageMakerEndpointTastSet(TaskSet):

    def __init__(self, parent):
        super().__init__(parent)
        self.config = SageMakerConfig()

    @task
    def test_invoke(self):
        # Start run here
        region = self.client.base_url.split("://")[1].split(".")[2]

        sagemaker_client = boto3.client('sagemaker-runtime', region_name=region, endpoint_url=self.client.base_url)

        # load image
        img_name = os.path.join(os.path.dirname(__file__), self.config.data)
        with open(img_name, "rb") as f:
            image_bytes = f.read()

        response = sagemaker_client.invoke_endpoint(
            EndpointName=self.config.endpointname,
            Body=image_bytes,
            ContentType='application/binary',
            Accept='application/json'
        )

        body = response["Body"].read()

        print(body)


class SageMakerEndpointLocust(HttpLocust):
    task_set = SageMakerEndpointTastSet
    min_wait = 5000
    max_wait = 15000
