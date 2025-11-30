# 📄 Chase Policy RAG Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-v0.3-green?style=for-the-badge&logo=langchain&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-black?style=for-the-badge&logo=ollama&logoColor=white)

A local **Retrieval-Augmented Generation (RAG)** application that allows users to chat with the **Chase Sapphire Credit Card Policy** document.

Built with **LangChain**, **Ollama (Llama 3)**, and **Streamlit**, this project runs entirely offline, ensuring data privacy while delivering accurate, context-aware answers using advanced RAG techniques like **Query Expansion** and **Cross-Encoder Re-ranking**.

---

## 🚀 Features

* **🔒 Fully Local:** Runs on your machine using Ollama (no OpenAI API keys required).
* **🧠 Advanced Retrieval:**
    * **Query Expansion:** Generates sub-queries to find hidden details.
    * **Hybrid Search:** Uses semantic similarity to find relevant policy sections.
    * **Re-Ranking:** Uses a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) to grade the top results for maximum accuracy.
* **💬 Interactive UI:** Simple, chat-like interface built with Streamlit.
* **📄 PDF Support:** Upload and process any PDF document instantly.

---

## 🛠️ Architecture

The pipeline consists of the following steps:

1.  **Ingestion:** The PDF is loaded and split into small, semantically meaningful chunks using `SentenceTransformersTokenTextSplitter`.
2.  **Embedding:** Text chunks are converted into vectors using the `all-MiniLM-L6-v2` model and stored in **ChromaDB**.
3.  **Retrieval (The "Smart" Part):**
    * The user's question is expanded into multiple sub-queries.
    * The system retrieves the top 10 most relevant chunks.
    * A **Cross-Encoder** re-ranks these chunks to select the top 5 absolute best matches.
4.  **Generation:** The top 5 chunks are passed to **Llama 3**, which generates a natural language answer based *strictly* on the policy text.

---

## 💻 Installation & Setup

### 1. Prerequisites
* **Python 3.10+** installed.
* **Ollama** installed and running.
    * Download from [ollama.com](https://ollama.com).
    * Pull the Llama 3 model:
        ```bash
        ollama pull llama3
        ```

### 2. Clone the Repository
```bash
git clone [https://github.com/your-username/chase-policy-rag.git](https://github.com/your-username/chase-policy-rag.git)
cd chase-policy-rag