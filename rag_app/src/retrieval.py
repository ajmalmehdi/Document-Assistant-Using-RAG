from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

# Define the template for the prompt that will be used to query the LLM
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
Answer the question based on the context above: {question}
"""

def query_rag(query_text, chroma_path, embedding_model, llm_model):
    """
    Query the RAG (Retrieval-Augmented Generation) system to answer a question based on retrieved context.
    
    """
    # Initialize the Chroma database with the provided embedding model
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_model)
    
    # Retrieve the top 5 most relevant documents based on the query text
    results = db.similarity_search_with_score(query_text, k=5)
    
    # Concatenate the content of the top documents to form the context for the LLM
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
    # Create a prompt using the context and the question
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Generate the response from the LLM based on the formatted prompt
    response_text = llm_model.invoke(prompt)
    
    # Retrieve the sources (IDs) of the relevant documents used for the answer
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    
    # Format the final response with the answer and sources
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    
    # Print and return the response
    print(formatted_response)
    return response_text
