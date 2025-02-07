import ray
from get_links import get_links
from pyarrow import fs

import pytest
from testcontainers.localstack import LocalStackContainer
import os
from dotenv import load_dotenv

load_dotenv()

# Create Localstack test container to store documents in for testing
localstack = LocalStackContainer(image="localstack/localstack:4.1", region_name="us-west-1").with_bind_ports(4566, 4567)

AWS_ACCESS_KEY_ID = "test"
AWS_SECRET_ACCESS_KEY = "test"
AWS_ENDPOINT_URL = "http://localhost:4567"
API_TOKEN = os.getenv("API_TOKEN")

@pytest.fixture(scope="module", autouse=True)
def setup(request):
    localstack.start() # Start S3 container

    def remove_container():
        localstack.stop()

    request.addfinalizer(remove_container) # Stop the container after tests are over

    client = localstack.get_client(
        's3',
    )
    # Create a bucket to store documents in
    client.create_bucket(
        Bucket="documents",
        CreateBucketConfiguration={
            'LocationConstraint': 'us-west-1'
        }
    )

# Runs before each test
@pytest.fixture(scope="function", autouse=True)
def setup_bucket():
    client = localstack.get_client('s3')
    # Delete and recreate the bucket to clear everything
    client.delete_bucket(Bucket="documents")
    client.create_bucket(
        Bucket="documents",
        CreateBucketConfiguration={
            'LocationConstraint': 'us-west-1'
        }
    )


def test_get_links():
    runtime_env = {
        "env_vars": {
            "AWS_ACCESS_KEY_ID": AWS_ACCESS_KEY_ID,
            "AWS_SECRET_ACCESS_KEY": AWS_SECRET_ACCESS_KEY,
            "AWS_ENDPOINT_URL": AWS_ENDPOINT_URL,
            "API_TOKEN": API_TOKEN
        } 
    }

    ds = ray.get(get_links.options(runtime_env=runtime_env).remote()).materialize()

    assert "link" in ds.columns()
    assert ds.count() > 0

    # Write the dataset to S3
    filesys = fs.S3FileSystem(endpoint_override=AWS_ENDPOINT_URL)
    ds.write_parquet("s3://documents/", filesystem=filesys)

    # Run get_links, this time testing functionality for when documents are already in S3
    new_ds = ray.get(get_links.options(runtime_env=runtime_env).remote()).materialize()

    # The second run of get_links shouldn't have added any new documents
    assert "link" in new_ds.columns()
    assert new_ds.count() < ds.count()
