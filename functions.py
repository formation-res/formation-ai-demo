#pip install --upgrade langchain langchain-community langchainhub gpt4all langchain-openai chromadb bs4 sentence-transformers
# pip install jq
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
# import chromadb
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_community.document_loaders import JSONLoader
import json
from pathlib import Path
from pprint import pprint
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

# set Embedding to all-MiniLM-L6-v2
def setEmbeddingMini():
    ef = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return ef

# set Embedding to OPEN AI Embeddings
def setEmbeddingOpenAI():
    ef = OpenAIEmbeddings()
    return ef

import os
from dotenv import load_dotenv
load_dotenv()

# USE: load Data from the Web
def webLoader(link):
    loader = WebBaseLoader(link)
    data = loader.load()
    return data

# USE: split data for vector storage
def splitData(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50, add_start_index=True)
    all_splits = text_splitter.split_documents(data)
    return all_splits

# USE: create vector store
# currently using embedding all-MiniLM-L6-v2
def createVectorStore(data, embedding_function):
    
    #filter metadata that may lead to errors (lists, for example, will not be parsed as meta data)
    data = filter_complex_metadata(data)
    vectorstore = Chroma.from_documents(documents=data, embedding=embedding_function)
    
    vectorstore
    return vectorstore

# USE: set Gemma as LLM
def setGemma():
    llm = Ollama(
    model="gemma:2b", 
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0.1  # Set temperature to a low value, adjust as needed
    )
    return llm

# USE: set Llama2 as LLM
def setLlama2():
    llm = Ollama(
        model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        temperature=0.1  # Set temperature to a low value, adjust as needed
        )
    return llm

# USE: set GPT4 as LLM
def setGPT4():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return llm

def similaritySearch(vectorstore, k):
    # default search, returns top k = 4 
    #retriever_simple = vectorstore.similarity_search(question)
    
    # provides the possibility to set top k value 
    retriever_modded = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    #another method to retrive, based on score threshold
    #retriever_modded = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
    
    return retriever_modded
    
# USE: joins the results into one doc for the llm to use as context
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# USE: load JSON data
# param: 
#   jq_schema : points to inner schema to accquire data from
#   content_key : points to specific entity in JSON file to be used as context
def jsonLoader(file_path, jq_schema, content_key):
    loader = JSONLoader(
    file_path=file_path,
    jq_schema=jq_schema,
    content_key=content_key,
    is_content_key_jq_parsable=True,
    )

    data = loader.load()
    return data



