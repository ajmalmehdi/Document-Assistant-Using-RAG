import argparse
import asyncio
from src.data import load_documents, split_documents, clear_database
from src.embeddings import add_to_chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

# Constants for file paths
CHROMA_PATH = "chroma"  # Path to the Chroma database
DATA_PATH = "data/data.txt"  # Path to the data file containing documents

def main():
    """
    Main function to handle arguments, load documents, process them, and 
    store their embeddings in the Chroma database.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")  # Optional flag to reset the database
    args = parser.parse_args()
    
    # Reset the database if the '--reset' flag is provided
    if args.reset:
        print("Clearing Database")  # Inform the user about database clearing
        clear_database()  # Clear the existing database

    # Load documents from the specified data file
    documents = load_documents(DATA_PATH)
    # Split the documents into smaller chunks
    chunks = split_documents(documents)
    # Initialize the embedding model (Ollama's embedding model)
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    # Asynchronously add the document chunks and their embeddings to the Chroma database
    asyncio.run(add_to_chroma(chunks, CHROMA_PATH, embedding))

# Entry point of the script
if __name__ == "__main__":
    main()
