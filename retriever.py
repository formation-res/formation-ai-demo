#pip install lark
import pprint
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
import functions    

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# llm = functions.setLlama2()
llm = OpenAI(temperature=0)
#embedding_function = functions.setEmbeddingMini()
embedding_function = OpenAIEmbeddings()
data = functions.jsonLoader("geo_Mod.json", ".")
data = functions.splitData(data)
vectorstore = functions.createVectorStore(data, embedding_function)
    

metadata_field_info = [
    AttributeInfo(
        name="seq_num",
        description="The sequence number of the point",
        type="integer",
    ),
    AttributeInfo(
        name="coordinates",
        description="The coordinates of the point",
        type="list[integer]",
    ),
    AttributeInfo(
        name="id",
        description="The ID number attributed to the point",
        type="integer",
    ),
    AttributeInfo(
        name="source",
        description="The file path of the point",
        type="string",
    ),
]
document_content_description = "Points of attraction, their id, and their coordinates"
retriever = SelfQueryRetriever.from_llm(
    llm, vectorstore, document_content_description, metadata_field_info, verbose=True
)


pprint.pprint(retriever.get_relevant_documents("hi"))
# pprint.pprint(data)
# Chaindata = functions.splitData(data)
    
# vectorstore = functions.createVectorStore(data, embedding_function)
# len(vectorstore)


# chain = {"docs": functions.format_docs} | prompt | llm | StrOutputParser()
