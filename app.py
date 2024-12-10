import streamlit as st
import argparse
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from langchain_chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
import asyncio

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---
Answer the question based on the context above: {question}
"""

# Function to query the RAG system
async def query_rag(query_text: str):
    embedding = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://ollama:11434")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    results = db.similarity_search_with_score(query_text, k=5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    model = Ollama(model="gemma2", base_url="http://ollama:11434")
    response_text = model.invoke(prompt)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return formatted_response

# Streamlit app to create the chatbot interface
def main():
    st.title("Document-based Chatbot")

    # Initialize the chat history (messages)
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Chat input
    user_input = st.text_input("Ask a question:")

    # Display chat history
    for message in st.session_state.messages:
        st.markdown(f"**{message['role']}:** {message['content']}")

    # When the user submits a question
    if user_input:
        # Add user message to chat history
        st.session_state.messages.append({"role": "User", "content": user_input})

        # Query the RAG system asynchronously
        response = asyncio.run(query_rag(user_input))

        # Add the chatbot's response to the chat history
        st.session_state.messages.append({"role": "Bot", "content": response})

        # Clear the input field and rerun the app
        st.rerun()

if __name__ == "__main__":
    main()
