from langchain_chroma import Chroma
from langchain.schema import Document
import asyncio

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

async def add_to_chroma(chunks, chroma_path, embedding_model):
    embedding = embedding_model
    texts = [chunk.page_content for chunk in chunks]
    embeddings = await embedding.aembed_documents(texts)

    db = Chroma(
        persist_directory=chroma_path,
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
