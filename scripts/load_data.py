import argparse
import asyncio
from src.data import load_documents, split_documents, clear_database
from src.embeddings import add_to_chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "data/data.txt"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    documents = load_documents(DATA_PATH)
    chunks = split_documents(documents)
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    asyncio.run(add_to_chroma(chunks, CHROMA_PATH, embedding))

if __name__ == "__main__":
    main()
