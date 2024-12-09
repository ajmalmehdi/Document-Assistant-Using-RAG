from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
Answer the question based on the context above: {question}
"""

def query_rag(query_text, chroma_path, embedding_model, llm_model):
    db = Chroma(persist_directory=chroma_path, embedding_function=embedding_model)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    response_text = llm_model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text
