import ray

from convert_documents import convert_documents
from index_documents import index_documents

from dotenv import load_dotenv

load_dotenv()

ray.init()

####### Document Conversion #######

# Get metadata from Digital Commons and convert the PDFs
convert_documents()

####### Indexing #######

# index_documents()
