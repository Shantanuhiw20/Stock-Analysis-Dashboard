# rag_utils.py
import fitz  # PyMuPDF
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from utils.config import LLM_MODEL
import tensorflow as tf
from langchain.chains.retrieval_qa.base import RetrievalQA


def extract_text_from_pdf_stream(uploaded_file) -> str:
    """
    Reads a PDF file from a stream and extracts all text content.
    """
    text = ""
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text


def build_vector_store(texts: List[str], persist_dir: str = None) -> Chroma:
    """
    Splits texts into chunks, embeds them, and builds a Chroma vector store.
    Optionally persists the store to disk.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = []
    for text in texts:
        docs.extend(splitter.create_documents([text]))
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    if persist_dir:
        db = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_dir)
        db.persist()
    else:
        db = Chroma.from_documents(documents=docs, embedding=embeddings)
    return db


def load_chroma_db(uploaded_2022_23, uploaded_2023_24, persist_dir: str = None) -> Chroma:
    """
    Loads or builds a Chroma vector store from two uploaded PDF files.
    """
    if persist_dir:
        return Chroma(
            persist_directory=persist_dir,
            embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2"),
        )
    text_22 = extract_text_from_pdf_stream(uploaded_2022_23)
    text_23 = extract_text_from_pdf_stream(uploaded_2023_24)
    return build_vector_store([text_22, text_23], persist_dir=persist_dir)


def create_qa_chain(db: Chroma, k: int = 3) -> RetrievalQA:
    """
    Creates a RetrievalQA chain using the provided Chroma vector store.
    """
    # Clear any existing Keras session to free resources
    tf.keras.backend.clear_session()
    retriever = db.as_retriever(search_kwargs={"k": k})
    llm = ChatGroq(model_name=LLM_MODEL, temperature=0.0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=False
    )
    return chain


def chat_with_rag(db: Chroma, user_question: str, k: int = 3) -> str:
    """
    Runs a user query through the RetrievalQA chain and returns the answer.
    """
    chain = create_qa_chain(db, k=k)
    return chain.run({"query": user_question})
