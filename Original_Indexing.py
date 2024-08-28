from pathlib import Path

import qdrant_client
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.llms.ollama import Ollama
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.settings import Settings

# Path to your data folder
documents = SimpleDirectoryReader("docs").load_data()

# Initialize the vector store
client = qdrant_client.QdrantClient(
    path="../qdrant_data"
)
vector_store = QdrantVectorStore(client=client, collection_name="data2")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Settings for llm (LlamaIndex 0.10)
Settings.llm = Ollama(model="mistral:instruct", embed_model="local", request_timeout=2000, gpu_memory_utilization=0.5)
Settings.embed_model = "local"

# Indexing
index = VectorStoreIndex.from_documents(documents)

# If this throws an OutOfMemoryError, adjust gpu_memory_utilization in settings

# Querying
query_engine = index.as_query_engine()

# Sending the query
response = query_engine.query("What is the document about?")

# Printing the response
print(response)