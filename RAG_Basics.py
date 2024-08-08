from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.llms.ollama import Ollama
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.settings import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index.core.node_parser import SentenceSplitter

import torch
torch.cuda.empty_cache()

documents = SimpleDirectoryReader("docs").load_data()

text_splitter = SentenceSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        )

chunked_documents = text_splitter.get_nodes_from_documents(documents)

embed_model = "BAAI/bge-small-en-v1.5"
#    gpt4all_kwargs = {'allow_download': 'True'} #For Edouard: This part needed in tutorial


chroma_client = chromadb.PersistentClient(path="persist")

chroma_client.delete_collection("quickstart") # If re-indexing
chroma_collection = chroma_client.create_collection("quickstart")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


Settings.llm = Ollama(model="mistral:instruct", embed_model="local", request_timeout=2000, gpu_memory_utilization=0)
Settings.embed_model = HuggingFaceEmbedding("local", model_name = embed_model, embed_batch_size=100)

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context, embed_model=embed_model
)

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

response = query_engine.query("What is the document about?")

print(response)
