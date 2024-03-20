# Demo: Locally running LLM w/ RAG framework

This project showcases locally run LLM's with a RAG framework built on top.

The purpose of this demo is to test how locally run LLM's may deal with the typical datasets Formation GmbH works with. 

In the demo, the LLM is provided "context" relevant to Formation GmbH through the framework of [***Retrieval Augmented Generation***](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/) (RAG); the first being webscraped data, and the second being JSON files.

The demo currently is set to showcase two locally run models (using Ollama) Gemma 2 and Llama 2, and can be compared to OPENAI models.

Currently, the embeddings used is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), using Chroma

# Installation

    pip install --upgrade langchain langchain-community langchainhub gpt4all langchain-openai chromadb bs4 sentence-transformers jq

To test with OPENAI models, and to use [LangSmith](https://www.langchain.com/langsmith) to trace the code, create a .env file with their respective API Keys.

# Usage

    run `main.py`

modify functions `loadData()`, `loadLLM()`, `loadEmbedding()` to choose what Data/LLM/Embedding to run with. 
Other models can be easily added.

`functions.py` contains functions seperating every step, where one may understand the workflow with ease.

`json_modifier_*.py` are scripts that help modify JSON files to ensure that correct parts of the file are passed in as context for RAG. 
Through testing and tracing, I realized meta-data isn't passed in as context and therefore created a new variable that stores all data fields needed as one string, which is then passed in.

Please look at `geo.json` and `geo_Mod.json` as simple examples.

`Formation1.json` is a JSON file typical to Formation's use case. `Formation1_Mod.json` is the modified json I use as input where variable `page_content` contains the context passed into the RAG.



