import ray
from ray import workflow
import ray.data

import logging

from get_links import get_links
from convert_documents import convert_documents
from index_documents import index_documents 

import json

ray.init()

# Get runtime environment to pass to workflow tasks
runtime_env = json.loads(ray.get_runtime_context().get_runtime_env_string())

####### Document Conversion Workflow #######

# Define first step of workflow - getting the links
links = get_links.options(runtime_env=runtime_env).bind()
# Pass the links to convert them to documents
documents = convert_documents.options(runtime_env=runtime_env).bind(links)

workflow.run(documents)

####### Indexing Workflow #######

# Pass the links to the indexing step of the workflow
documents = index_documents.options(runtime_env=runtime_env).bind()

workflow.run(documents)