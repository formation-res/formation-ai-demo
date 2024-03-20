# Demo: Locally running LLM w/ RAG framework

This project showcases locally run LLM's with a RAG framework built on top.

The purpose of this demo is to test how locally run LLM's may deal with the typical datasets Formation GmbH works with. 

In the demo, the LLM is provided "context" relevant to Formation GmbH through the framework of [***Retrieval Augmented Generation***](https://ai.meta.com/blog/retrieval-augmented-generation-streamlining-the-creation-of-intelligent-natural-language-processing-models/) (RAG); the first being webscraped data, and the second being JSON files.

The demo currently is set to showcase two locally run models (using Ollama) Gemma 2 and Llama 2, and can be compared to OPENAI models.

Currently, the embeddings used is [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
