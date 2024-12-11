from langchain_chroma import Chroma
from langchain.schema import Document
import asyncio

def calculate_chunk_ids(chunks):
    """
    Calculate and assign unique IDs for each chunk based on source and page.
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        
        # Increment chunk index for same page or reset for new page
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        
        # Assign chunk ID based on page and chunk index
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

async def add_to_chroma(chunks, chroma_path, embedding_model):
    """
    Add document chunks to the Chroma database with embeddings.
    """
    # Generate embeddings for the document chunks
    embedding = embedding_model
    texts = [chunk.page_content for chunk in chunks]
    embeddings = await embedding.aembed_documents(texts)

    # Initialize Chroma DB connection
    db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding  
    )
    
    # Calculate unique IDs for the chunks
    chunks_with_ids = calculate_chunk_ids(chunks)
    
    # Retrieve existing items in the database to avoid duplicates
    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    new_chunks = []
    new_embeddings = []
    
    # Collect new chunks that are not in the DB
    for chunk, embedding in zip(chunks_with_ids, embeddings):
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            new_embeddings.append(embedding)

    # Add new documents to the DB if any
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
