from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
import os

# Create embeddings model (runs locally)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def load_and_split_documents(file_path: str) -> List[Document]:
    """Load a document(PDF or TXT) and split it into chunks."""
    # Choose the correct loader based on file type
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    # Load the document
    documents = loader.load()

    # Split into small chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # Maximum size of each chunk
        chunk_overlap=200,    # Overlap between chunks
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks: List[Document], persist_directory: str = "./chroma_db"):
    """Create a Chroma vector store from document chunks."""
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def get_retriever(persist_directory: str = "./chroma_db", k: int = 3):
    """Load an existing vector store and return a retriever."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})

def add_documents_to_store(chunks: List[Document], persist_directory: str = "./chroma_db"):
    """Add new chunks to the existing Chroma store."""
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)
    return vectorstore