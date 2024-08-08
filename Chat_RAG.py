import os
from dotenv import load_dotenv

import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
#from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
import chromadb

from langchain_community.embeddings import GPT4AllEmbeddings 
from langchain_community.chat_models import ChatOllama


load_dotenv()


def load_chunk_persist_pdf() -> Chroma:
    pdf_folder_path = "C:\\Users\\ZANAZZI\\GitHub\\Indexing\\docs"
    documents = []
    for file in os.listdir(pdf_folder_path):
        if file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder_path, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunked_documents = text_splitter.split_documents(documents)
    client = chromadb.Client()
    if client.list_collections():
        consent_collection = client.create_collection("consent_collection")
    else:
        print("Collection already exists")
    model_name_embed = "all-MiniLM-L6-v2.gguf2.f16.gguf"
    gpt4all_kwargs = {'allow_download': 'True'} #For Edouard: This part needed in tutorial
    embeddings_bert = GPT4AllEmbeddings(
    model_name = model_name_embed,
    gpt4all_kwargs=gpt4all_kwargs
    )
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings_bert,
        persist_directory="persist"
    )
    vectordb.persist()
    return vectordb


def create_agent_chain():
    model_name = "mistral:latest"
    llm = ChatOllama(model=model_name, temperature=0.2)
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
st.header("Query PDF Source")

form_input = st.text_input('Enter Query')
submit = st.button("Generate")

if submit:
    st.write(get_llm_response(form_input))