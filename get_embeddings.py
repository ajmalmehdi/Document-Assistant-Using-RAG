from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings.ollama import OllamaEmbeddings
import asyncio
from langchain.schema import Document

async def get_embedding_function():
    texts = [chunk.page_content if isinstance(chunk, Document) else chunk for chunk in chunks]
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return await embeddings.aembed_documents(texts)

