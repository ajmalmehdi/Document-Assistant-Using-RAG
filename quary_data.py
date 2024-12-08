from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
import argparse

from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
import asyncio
from get_embeddings import get_embedding_function



CHROMA_PATH = "chroma"
DATA_PATH = "data/CursusINPT.pdf"

PROMPT_TEMPLATE = """
Répondez à la question en vous basant uniquement sur le contexte suivant:

{context}

---
repodeez à la question on ce basant sur le contexte ci-dessus: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    results = db.similarity_search_with_score(query_text, k=10)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = Ollama(model="gemma2")
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    main()