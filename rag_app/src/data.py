from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import shutil
import os

# Constant for the Chroma database path
CHROMA_PATH = "chroma"

def load_documents(file_path):
    #Load documents from the text file. 

    # Initialize the TextLoader to load the file
    loader = TextLoader(file_path)
    # Load and return the documents
    return loader.load()

def split_documents(documents):
 
    #Split the loaded documents into smaller chunks for processing.

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  # Maximum size of each chunk
        chunk_overlap=80,  # Overlap between chunks to preserve context
        length_function=len,  # Length function to calculate chunk size
        is_separator_regex=False,  # Whether the separator is a regular expression
    )
    # Split and return the chunks
    return text_splitter.split_documents(documents)

def clear_database():
    # Check if the Chroma database folder exists
    if os.path.exists(CHROMA_PATH):
        # Delete the folder and all its contents
        shutil.rmtree(CHROMA_PATH)

