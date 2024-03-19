#pip install --upgrade langchain langchain-community langchainhub gpt4all langchain-openai chromadb bs4 sentence-transformers
# pip install jq
# install ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from pprint import pprint
from langchain_core.prompts import PromptTemplate
import functions   

import os
from dotenv import load_dotenv
load_dotenv()


# USE: Function to load data from a web source
def loadData():
    # Load data from JSON. 
    # two examples provided, 
    #   Formation1_Mod.json contains json formar pertaining to real use case
    #   geo_E2.json is a simpler generic example
    
    data_load = functions.jsonLoader("JSON/Formation_JSON/Formation1_Mod.json", ".hits[].hit", ".page_content")
    #data_load = functions.jsonLoader("JSON/Generic_JSON/geo_Mod.json", ".features[]", ".properties")
    
    
    # Load data from a web source using a custom function
    #data_load = functions.webLoader("https://www.jillesvangurp.com")
    
    return data_load

# Set up LLM 
def loadLLM():
    llm = functions.setLlama2()
    #llm = functions.setGemma()
    #llm = functions.setGPT4()
    return llm

# Set up Embedding function
def loadEmbedding():
    embedding_function = functions.setEmbeddingMini()
    #embedding_function = functions.setEmbeddingOpenAI()
    return embedding_function

# Set Question to be asked
def loadQuestion():
    question = "What is Formation GmbH?"
    return question

# Template for generating prompts
template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Try and keep the answer direct and consise
Always say "thanks for asking!" or something similar to be friendly at the end of the answer.

{context}

Question: {question}

Answer:"""

# Create a prompt template from the provided template
prompt_template= PromptTemplate.from_template(template=template)

llm = loadLLM()
embedding_function = loadEmbedding()

# Load data
data_load = loadData()
# Split data
data = functions.splitData(data_load)
# Create vector store
vectorstore = functions.createVectorStore(data, embedding_function)


# question
question= loadQuestion()

# Perform similarity search
# param: 
#   vectorstore: send in vectorstore
#   k : retrives the top k results of a given query after similarity search
docs = functions.similaritySearch(vectorstore=vectorstore,k = 4)


# TEST: Print the documents retrieved after similarity search on the
#pprint(docs.invoke(question))

# Chain of operations for generating a response
chain = (
    {"context": docs | functions.format_docs, "question": RunnablePassthrough()} 
    | prompt_template 
    | llm 
    | StrOutputParser()
)

# Ollama framework and OPENAI's framework have slightly different methods of printing. 
# Ollama prints automatically, OPENAI needs print to be called
# This try and catch block prevents Ollama's results to be printed twice.
try:
    if llm.model_name == "gpt-4":
        print(chain.invoke(question))
    else:
        chain.invoke(question)
except AttributeError:
    chain.invoke(question)
