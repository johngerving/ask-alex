import ray
from get_links import get_links

import pytest
from testcontainers.localstack import LocalStackContainer
import boto3
import os
from dotenv import load_dotenv

load_dotenv()

localstack = LocalStackContainer(image="localstack/localstack:4.1", region_name="us-west-1").with_bind_ports(4566, 4567)

@pytest.fixture(scope="module", autouse=True)
def setup(request):
    localstack.start()

    def remove_container():
        localstack.stop()

    request.addfinalizer(remove_container)

    client = localstack.get_client(
        's3',
    )
    client.create_bucket(
        Bucket="documents",
        CreateBucketConfiguration={
            'LocationConstraint': 'us-west-1'
        }
    )

def test_get_links():
    runtime_env = {
        "env_vars": {key: os.getenv(key) for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_ENDPOINT_URL", "API_TOKEN"]}
    }

    ds = ray.get(get_links.options(runtime_env=runtime_env).remote()).materialize()

    assert "link" in ds.columns()
    assert ds.count() > 0