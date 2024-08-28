import os
from dotenv import load_dotenv

import streamlit as st

#from langchain.document_loaders import PyPDFLoader
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.chains.question_answering import load_qa_chain
#from langchain.vectorstores import Chroma

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader
)
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.settings import Settings
import chromadb


#from langchain_community.embeddings import GPT4AllEmbeddings 
#from langchain_community.chat_models import ChatOllama

import torch
torch.cuda.empty_cache()

load_dotenv()


def load_chunk_persist_pdf() -> Chroma:
    documents = SimpleDirectoryReader("docs").load_data()
    text_splitter = SentenceSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        )
    chunked_documents = splitter.get_nodes_from_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    model_name_embed = "BAAI/bge-small-en-v1.5"
#    gpt4all_kwargs = {'allow_download': 'True'} #For Edouard: This part needed in tutorial
    embeddings_bert = HuggingFaceEmbedding(
    model_name = model_name_embed
    )
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings_bert,
        persist_directory="persist"
    )
    vectordb.persist()
    return vectordb


def create_agent_chain():
    model_name = "mistral:instruct"
    llm = Ollama(model=model_name, temperature=0.2, request_timeout = 500, gpu_memory_utilization=0.5)
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain


def get_llm_response(query):
    vectordb = load_chunk_persist_pdf()
    chain = create_agent_chain()
    matching_docs = vectordb.similarity_search(query)
    answer = chain.run(input_documents=matching_docs, question=query)
    return answer


# Streamlit UI
# ===============
st.set_page_config(page_title="Doc Searcher", page_icon=":robot:")
st.header("Ask me about the document")

instructions = st.text_input('How should I answer?')

form_input = st.text_input('What do you want to know?')

full_input = instructions + form_input

submit = st.button("Generate")

if submit:
    st.write(get_llm_response(full_input))