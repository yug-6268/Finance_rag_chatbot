# Finance RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for finance and chartered accountancy information. This chatbot uses LangChain, Groq API, and FAISS vector database to provide accurate answers based on financial documentation.

## Features
- Answers questions about financial regulations, taxes, and accounting standards
- Uses RAG to provide context-aware, accurate responses
- Built with LangChain, FAISS, and Streamlit

## Setup Instructions
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Unix)
4. Install dependencies: `pip install -r requirements.txt`
5. Create a .env file with your API keys: `GROQ_API_KEY=your_api_key_here`
6. Add your PDFs to the data folder
7. Run the application: `python run.py`
