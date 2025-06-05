import ray

from convert_documents import convert_documents
from index_documents import index_documents
from metadata_extraction.extract_metadata import extract_metadata

ray.init()

####### Document Conversion #######

# Get metadata from Digital Commons and convert the PDFs
# convert_documents()

####### Metadata Extraction #######
# extract_metadata()

####### Indexing #######

index_documents()
