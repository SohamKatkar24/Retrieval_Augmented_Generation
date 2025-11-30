import streamlit as st
import os
import tempfile
import time

# --- LangChain & RAG Imports ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# --- CONFIGURATION ---
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📄 PDF Policy Chatbot (Local RAG)")

# --- CACHED RESOURCES (Run once) ---
@st.cache_resource
def load_models():
    """Load heavy models once to speed up the app."""
    embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatOllama(model="llama3", temperature=0)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return embedding_func, llm, cross_encoder

embedding_function, llm, cross_encoder = load_models()

# --- HELPER FUNCTIONS ---
def process_pdf(uploaded_file):
    """Save upload to temp file and build vector DB."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    raw_docs = loader.load()
    
    # Split
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    char_docs = char_splitter.split_documents(raw_docs)
    
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=50, tokens_per_chunk=256, model_name="all-MiniLM-L6-v2")
    final_docs = token_splitter.split_documents(char_docs)
    
    # Vector DB
    db = Chroma.from_documents(final_docs, embedding_function, persist_directory="./chroma_db_streamlit")
    return db

def expand_query(query):
    """Generate sub-queries to improve search."""
    template = """Break down this query into 2 simple sub-queries for a policy document search. 
    Query: {query} 
    Output only the sub-queries, separated by newlines."""
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": query}).strip().split("\n")

def rerank_docs(query, docs, top_k=5):
    if not docs: return []
    pairs = [[query, doc.page_content] for doc in docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]

# --- SIDEBAR: FILE UPLOAD ---
with st.sidebar:
    st.header("1. Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file and "db_ready" not in st.session_state:
        with st.status("Processing PDF...", expanded=True) as status:
            st.write("Reading file...")
            db = process_pdf(uploaded_file)
            st.session_state.db = db
            st.session_state.db_ready = True
            status.update(label="✅ PDF Processed!", state="complete", expanded=False)

# --- MAIN CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your PDF..."):
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Generate Response
    if "db_ready" not in st.session_state:
        with st.chat_message("assistant"):
            st.error("⚠️ Please upload a PDF first!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # A. Search & Rerank
                subqueries = expand_query(prompt)
                search_queries = [prompt] + subqueries
                
                raw_docs = []
                for q in search_queries:
                    raw_docs.extend(st.session_state.db.similarity_search(q, k=3))
                
                # Deduplicate
                unique_docs = {d.page_content: d for d in raw_docs}.values()
                final_docs = rerank_docs(prompt, list(unique_docs), top_k=5)
                
                # B. Generate Answer
                context = "\n\n".join([d.page_content for d in final_docs])
                gen_template = """Answer the question based strictly on the context below.
                
                Context: {context}
                
                Question: {question}
                """
                chain = PromptTemplate.from_template(gen_template) | llm | StrOutputParser()
                response = chain.invoke({"context": context, "question": prompt})
                
                st.markdown(response)
                
                # Optional: Show sources in an expander
                with st.expander("View Source Context"):
                    for i, doc in enumerate(final_docs):
                        st.caption(f"**Source {i+1}:** {doc.page_content[:200]}...")

        # 3. Save Assistant Message
        st.session_state.messages.append({"role": "assistant", "content": response})