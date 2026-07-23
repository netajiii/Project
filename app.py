import streamlit as st
import os
import dotenv
import uuid
if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, AIMessage
from rag_methods import (
    load_doc_to_db,
    load_url_to_db,
    stream_llm_response,
    stream_llm_rag_response,
)

dotenv.load_dotenv()
os.environ["USER_AGENT"] = "myagent"

st.set_page_config(
    page_title="Local RAG LLM App",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Header ---
st.html("""<h2 style="text-align: center;">📚🔍 Local RAG Chatbot (Ollama) 🤖💬</h2>""")

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! Upload documents or add URLs and ask me anything."}
    ]

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    st.toggle("Use RAG (Document Search)", value=True, key="use_rag")
    
    st.divider()
    st.header("RAG Sources")
    st.file_uploader(
        "📄 Upload documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        on_change=load_doc_to_db,
        key="rag_docs",
    )
    st.text_input(
        "🌐 Add Website URL",
        placeholder="https://example.com",
        on_change=load_url_to_db,
        key="rag_url",
    )
    
    with st.expander("📚 Loaded Documents"):
        st.write(st.session_state.rag_sources if st.session_state.rag_sources else "No documents loaded yet.")

    st.divider()
    st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

# --- Main Chat ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        llm_stream = ChatOllama(
            model="llama3.2",
            temperature=0.7,
            streaming=True,
        )
        messages = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in st.session_state.messages]
        
        if st.session_state.use_rag and "vector_db" in st.session_state:
            st.write_stream(stream_llm_rag_response(llm_stream, messages))
        else:
            st.write_stream(stream_llm_response(llm_stream, messages))
    
