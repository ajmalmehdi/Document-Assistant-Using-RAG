from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import argparse
import os
import shutil
import asyncio
from langchain_community.embeddings.ollama import OllamaEmbeddings

CHROMA_PATH = "chroma"
DATA_PATH = "C:\\Users\\Hp\\Desktop\\inpt scrap\\data\\data.txt"



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    asyncio.run(add_to_chroma(chunks))  

async def add_to_chroma(chunks: list[Document]):
    embedding = OllamaEmbeddings(model="mxbai-embed-large")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = await embedding.aembed_documents(texts)
    
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding  
    )
    
    chunks_with_ids = calculate_chunk_ids(chunks)
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    new_chunks = []
    new_embeddings = []
    for chunk, embedding in zip(chunks_with_ids, embeddings):
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            new_embeddings.append(embedding)

    if len(new_chunks):
        print(f"Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(
            documents=new_chunks,
            embeddings=new_embeddings,
            ids=new_chunk_ids
        )
    else:
        print("No new documents to add")

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def load_documents():
    loader = TextLoader("C:\\Users\\Hp\\Desktop\\inpt scrap\\data\\data.txt")
    return loader.load()

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
