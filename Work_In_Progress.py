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
def setEmbeddingMini():
    ef = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return ef

# USE: load Data from the Web
def webLoader(link):
    loader = WebBaseLoader(link)
    data = loader.load()
    return data

#
def splitData(data):
    #if exists, load ,else make 
    # if Text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50, add_start_index=True)
    all_splits = text_splitter.split_documents(data)

    #if JSON
    #currently not working.
    # pprint(data)
    # text_splitter = RecursiveJsonSplitter(max_chunk_size=1000)
    # all_splits = text_splitter.split_json(json_data=data)
    
    # print("all_splits length:", len(all_splits), "\n *********")
    # for data in all_splits:
    #     print (data.page_content,"\n")
    return all_splits

def createVectorStore(data, embedding_function):
    
    data = filter_complex_metadata(data)
    #pprint(data)
    vectorstore = Chroma.from_documents(documents=data, embedding=embedding_function)
    
    vectorstore
    return vectorstore

def setGemma():
    llm = Ollama(
        model="gemma:2b", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    return llm

def setLlama2():
    llm = Ollama(
        model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))
    return llm

def setGPT4():
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return llm

def similaritySearch(vectorstore, k):
    # Run
    #simple method, although only returns top k of 4
    # retriever_simple = vectorstore.similarity_search(question)
    # len(retriever_simple)
    
    #set number of Top K returns 
    retriever_modded = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    #another method to retrive, based on score threshold
    #retriever_modded = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
    
    # retrieved_docs = retriever_modded.invoke(question)
    # print(len(retrieved_docs))
    # pprint(retrieved_docs)
    return retriever_modded
    
    
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def jsonLoader(file_path, jq_schema, content_key):
    loader = JSONLoader(
    file_path=file_path,
    jq_schema=jq_schema,
    content_key=content_key,
    is_content_key_jq_parsable=True,
    )

    data = loader.load()
    return data


# Define the metadata extraction function.
def metadata_func(record: dict, metadata: dict) -> dict:

    metadata["coordinates"] = record.get("coordinates")
    metadata["id"] = record.get("id")

    if "source" in metadata:
        source = metadata["source"].split("/")
        source = source[source.index("GitHub"):]
        metadata["source"] = "/".join(source)

    return metadata

