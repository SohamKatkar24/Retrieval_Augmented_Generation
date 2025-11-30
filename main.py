import os
import pandas as pd

# Libraries for Loading & Splitting
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

# Libraries for Vector Store (Chroma) & Embeddings
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Libraries for RAG & Ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from sentence_transformers import CrossEncoder

# ==========================================
# PART 1: SETUP & CONFIGURATION
# ==========================================
PDF_FILE = "chase_policy.pdf"
DB_PATH = "./chroma_db"

print("--- STARTING RAG PIPELINE ---")

# ==========================================
# PART 2: LOAD & SPLIT DOCUMENT
# ==========================================
if os.path.exists(DB_PATH) and os.listdir(DB_PATH):
    print(f"Existing vector database found at {DB_PATH}. Skipping ingestion.")
    # Initialize Embedding Model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load existing DB
    db = Chroma(persist_directory=DB_PATH, embedding_function=embedding_function)
else:
    print("Loading and splitting PDF... (This may take a minute)")
    loader = PyPDFLoader(PDF_FILE)
    raw_documents = loader.load()

    # 1. Character Split
    character_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    character_split_docs = character_splitter.split_documents(raw_documents)

    # 2. Token Split (for better semantic search)
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=50, tokens_per_chunk=256, model_name="all-MiniLM-L6-v2")
    final_docs = token_splitter.split_documents(character_split_docs)

    # 3. Create Vector DB
    print("Creating Vector Database...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(final_docs, embedding_function, persist_directory=DB_PATH)
    print("Database created and saved.")

# ==========================================
# PART 3: INITIALIZE MODELS (OLLAMA & CROSS-ENCODER)
# ==========================================
print("Initializing Ollama (Llama3) and Cross-Encoder...")
llm = ChatOllama(model="llama3", temperature=0)
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# ==========================================
# PART 4: HELPER FUNCTIONS (EXPANSION & RERANKING)
# ==========================================
def generate_subqueries(original_query):
    template = """
    You are a helpful assistant. Break down the following query into 2-3 simpler 
    sub-queries for a credit card policy search.
    Query: {query}
    Output only the sub-queries, separated by newlines. Do not number them.
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"query": original_query})
        return response.strip().split("\n")
    except Exception as e:
        print(f"Error generating subqueries: {e}")
        return []

def generate_hypothetical_answer(original_query):
    template = """
    Provide a hypothetical, plausible answer to the question below regarding a 
    credit card policy. Focus on keywords.
    Question: {query}
    """
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"query": original_query})

def rerank_documents(query, retrieved_docs, top_k=5):
    if not retrieved_docs:
        return []
    pairs = [[query, doc.page_content] for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)
    scored_docs = list(zip(retrieved_docs, scores))
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs[:top_k]]

# ==========================================
# PART 5: MAIN PROCESSING FUNCTION
# ==========================================
def process_query(user_query):
    print(f"\nProcessing Query: '{user_query}'")
    
    # 1. Expand Query
    subqueries = generate_subqueries(user_query)
    hypothetical_ans = generate_hypothetical_answer(user_query)
    search_queries = [user_query] + subqueries + [hypothetical_ans]
    
    # 2. Retrieve Documents
    retrieved_docs = []
    for q in search_queries:
        docs = db.similarity_search(q, k=3)
        retrieved_docs.extend(docs)
        
    # Deduplicate
    unique_docs = {doc.page_content: doc for doc in retrieved_docs}.values()
    
    # 3. Re-rank
    final_docs = rerank_documents(user_query, list(unique_docs), top_k=5)
    
    # 4. Generate Answer
    if not final_docs:
        return "I could not find any relevant information."
        
    context_text = "\n\n".join([doc.page_content for doc in final_docs])
    
    generation_template = """
    Use the provided context to answer the user's question about the credit card policy.
    If the answer is not in the context, say you don't know.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    prompt = PromptTemplate.from_template(generation_template)
    chain = prompt | llm | StrOutputParser()
    final_answer = chain.invoke({"context": context_text, "question": user_query})
    
    return final_answer

# ==========================================
# PART 6: USER INTERFACE (LOOP)
# ==========================================
if __name__ == "__main__":
    print("\n✅ SYSTEM READY! Ask a question about the Chase Policy.")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("Your Question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        answer = process_query(user_input)
        print(f"\n🤖 ANSWER:\n{answer}\n")
        print("-" * 50)