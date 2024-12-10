import argparse
from src.retrieval import query_rag
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama

CHROMA_PATH = "chroma"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    model = Ollama(model="gemma2")
    query_rag(args.query_text, CHROMA_PATH, embedding, model)

if __name__ == "__main__":
    main()