# Machine Learning Chatbot

A chatbot built with Streamlit and LangChain to answer machine learning questions based on a provided PDF book.

## Features
- Answers machine learning-related questions using Retrieval-Augmented Generation (RAG).
- ChatGPT-like interface with the input box appearing below the latest response.
- Rejects non-machine learning questions with: "It is not related to Machine Learning."
- Built with Python, Streamlit, LangChain, and Hugging Face models.

## Setup
1. Install dependencies:
   ```bash
   pip install -U langchain langchain-community langchain-huggingface pypdf transformers torch sentence-transformers chromadb streamlit
