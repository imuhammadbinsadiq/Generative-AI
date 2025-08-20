import os
import re
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import torch

# Function to clean text for embedding
def clean_text(text):
    text = re.sub(r'[^\x20-\x7E\n\t]', '', text)  # Keep printable ASCII, newlines, tabs
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text if text else None

# Function to check if the query is related to machine learning
def is_ml_related(query):
    ml_keywords = [
        'machine learning', 'deep learning', 'neural network', 'regression', 'classification',
        'clustering', 'supervised learning', 'unsupervised learning', 'reinforcement learning',
        'feature engineering', 'model training', 'overfitting', 'underfitting', 'gradient descent',
        'decision tree', 'random forest', 'support vector machine', 'svm', 'natural language processing',
        'nlp', 'computer vision', 'data preprocessing', 'hyperparameter', 'cross-validation',
        'confusion matrix', 'accuracy', 'precision', 'recall', 'f1 score', 'loss function',
        'optimization', 'backpropagation', 'convolutional neural network', 'cnn',
        'recurrent neural network', 'rnn', 'transformer', 'attention mechanism'
    ]
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in ml_keywords)

# Load and process the machine learning book (PDF)
def load_and_split_document(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        valid_chunks = []
        for chunk in chunks:
            cleaned_text = clean_text(chunk.page_content)
            if cleaned_text and len(cleaned_text) > 0:
                chunk.page_content = cleaned_text
                valid_chunks.append(chunk)
            else:
                print(f"Skipped invalid chunk: {chunk.metadata}")
        st.info(f"Loaded and split document into {len(valid_chunks)} valid chunks.")
        return valid_chunks
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return []

# Create or load vector store for document embeddings
def create_vector_store(chunks, persist_directory="chroma_db"):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        # Check if vector store exists
        if os.path.exists(persist_directory):
            st.info("Loading existing vector store...")
            vector_store = Chroma(
                collection_name="ml_book_collection",
                embedding_function=embeddings,
                persist_directory=persist_directory
            )
        else:
            st.info("Creating new vector store...")
            batch_size = 32
            vector_store = None
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                if not vector_store:
                    vector_store = Chroma.from_documents(
                        documents=batch_chunks,
                        embedding=embeddings,
                        collection_name="ml_book_collection",
                        persist_directory=persist_directory
                    )
                else:
                    vector_store.add_documents(batch_chunks)
                st.info(f"Processed batch {i // batch_size + 1}/{len(chunks) // batch_size + 1}")
            vector_store.persist()
        st.success("Vector store ready.")
        return vector_store
    except Exception as e:
        st.error(f"Error creating/loading vector store: {e}")
        return None

# Initialize the language model
def initialize_llm():
    try:
        model_name = "google/flan-t5-base"  # More capable model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=200,
            device=0 if torch.cuda.is_available() else -1
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        return None

# Create conversational RAG chain
def create_rag_chain(vector_store, llm):
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        rag_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            memory=memory
        )
        return rag_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None

# Streamlit GUI
def main():
    st.set_page_config(page_title="Machine Learning Chatbot", page_icon="🤖", layout="wide")
    
    # Custom CSS for ChatGPT-like look
    st.markdown("""
        <style>
        .main { background-color: #ffffff; }
        .chat-container {
            max-width: 800px;
            margin: auto;
            padding: 20px;
        }
        .chat-history {
            max-height: 500px;
            overflow-y: auto;
            margin-bottom: 20px;
        }
        .chat-message {
            padding: 12px;
            border-radius: 10px;
            margin: 8px 0;
            max-width: 70%;
            font-size: 16px;
            line-height: 1.5;
        }
        .user-message {
            background-color: #007bff;
            color: white;
            margin-left: auto;
        }
        .bot-message {
            background-color: #f1f0f0;
            color: black;
            margin-right: auto;
        }
        .stTextInput > div > div > input {
            border-radius: 20px;
            padding: 12px;
            font-size: 16px;
            width: 100%;
            border: 1px solid #d1d1d1;
        }
        .stButton > button {
            border-radius: 20px;
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            margin-left: 10px;
        }
        .stForm {
            display: flex;
            align-items: center;
            width: 100%;
            max-width: 800px;
            margin: auto;
            padding: 10px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("Machine Learning Chatbot")
        st.markdown("""
        This chatbot answers questions about machine learning based on a provided PDF book.
        - Ask questions related to machine learning topics.
        - Non-machine learning questions will receive a response: "It is not related to Machine Learning."
        - The chatbot uses a Retrieval-Augmented Generation (RAG) model for accurate answers.
        """)
        st.markdown("---")
        st.subheader("Instructions")
        st.write("1. Ensure the `machine_learning_book.pdf` is in the project directory.")
        st.write("2. Type your question in the input box below the latest response and press 'Send' or Enter.")
        st.write("3. The chat history is displayed above the input.")
        st.markdown("---")
        st.caption("Built with Streamlit and LangChain")

    # Main content
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.title("🤖 Machine Learning Chatbot")
    st.markdown("Ask questions about machine learning, and I'll respond based on the provided book!")

    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
        pdf_path = "machine_learning_book.pdf"
        if not os.path.exists(pdf_path):
            st.error(f"Error: PDF file '{pdf_path}' not found.")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        # Initialize chatbot components
        with st.spinner("Loading and processing PDF..."):
            chunks = load_and_split_document(pdf_path)
            if not chunks:
                st.markdown('</div>', unsafe_allow_html=True)
                return

        with st.spinner("Creating vector store..."):
            vector_store = create_vector_store(chunks)
            if not vector_store:
                st.markdown('</div>', unsafe_allow_html=True)
                return

        with st.spinner("Initializing language model..."):
            llm = initialize_llm()
            if not llm:
                st.markdown('</div>', unsafe_allow_html=True)
                return

        with st.spinner("Setting up RAG chain..."):
            st.session_state.rag_chain = create_rag_chain(vector_store, llm)
            if not st.session_state.rag_chain:
                st.markdown('</div>', unsafe_allow_html=True)
                return

    # Chat history
    st.markdown('<div class="chat-history">', unsafe_allow_html=True)
    st.markdown("### Chat History")
    for index, (sender, message) in enumerate(st.session_state.chat_history):
        if sender == "You":
            st.markdown(f'<div class="chat-message user-message">{sender}: {message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{sender}: {message}</div>', unsafe_allow_html=True)

        # Place input form after the last response
        if index == len(st.session_state.chat_history) - 1:
            with st.form(key=f"query_form_{index}", clear_on_submit=True):
                st.markdown('<div class="stForm">', unsafe_allow_html=True)
                query = st.text_input("Your Question:", key=f"query_input_{index}", placeholder="Ask about machine learning...")
                submit_button = st.form_submit_button("Send")
                st.markdown('</div>', unsafe_allow_html=True)

                if submit_button and query:
                    # Display user message
                    st.session_state.chat_history.append(("You", query))
                    st.markdown(f'<div class="chat-message user-message">You: {query}</div>', unsafe_allow_html=True)

                    # Process query
                    if not is_ml_related(query):
                        response = "It is not related to Machine Learning."
                    else:
                        try:
                            with st.spinner("Generating response..."):
                                result = st.session_state.rag_chain.invoke({"question": query})
                                response = re.sub(r'\[.*?\]', '', result['answer']).strip()
                        except Exception as e:
                            response = f"Error generating response: {e}"

                    # Display bot response
                    st.session_state.chat_history.append(("Chatbot", response))
                    st.markdown(f'<div class="chat-message bot-message">Chatbot: {response}</div>', unsafe_allow_html=True)

    # Show input form if chat history is empty
    if not st.session_state.chat_history:
        with st.form(key="query_form_initial", clear_on_submit=True):
            st.markdown('<div class="stForm">', unsafe_allow_html=True)
            query = st.text_input("Your Question:", key="query_input_initial", placeholder="Ask about machine learning...")
            submit_button = st.form_submit_button("Send")
            st.markdown('</div>', unsafe_allow_html=True)

            if submit_button and query:
                # Display user message
                st.session_state.chat_history.append(("You", query))
                st.markdown(f'<div class="chat-message user-message">You: {query}</div>', unsafe_allow_html=True)

                # Process query
                if not is_ml_related(query):
                    response = "It is not related to Machine Learning."
                else:
                    try:
                        with st.spinner("Generating response..."):
                            result = st.session_state.rag_chain.invoke({"question": query})
                            response = re.sub(r'\[.*?\]', '', result['answer']).strip()
                    except Exception as e:
                        response = f"Error generating response: {e}"

                # Display bot response
                st.session_state.chat_history.append(("Chatbot", response))
                st.markdown(f'<div class="chat-message bot-message">Chatbot: {response}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()