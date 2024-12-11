import argparse
from src.retrieval import query_rag
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

# Constant for the Chroma database path
CHROMA_PATH = "chroma"

def main():
    """
    Main function to handle user input, query the RAG system, and return the result.
    """
    # Parse command-line arguments for the query text
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")  # The query that the user will input
    args = parser.parse_args()
    
    # Initialize the embedding model (Ollama's embedding model)
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    # Initialize the language model (Ollama's Gemma2 model)
    model = Ollama(model="gemma2")
    
    # Use the query_rag function to process the query and retrieve results
    query_rag(args.query_text, CHROMA_PATH, embedding, model)

# Entry point of the script
if __name__ == "__main__":
    main()
