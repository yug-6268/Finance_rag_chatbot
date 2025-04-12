import os

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

# Load .env variables
load_dotenv(find_dotenv())

# Groq API Key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192" 

# Step 1: Setup Groq LLM
def load_llm(model_name):
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name=model_name,
        temperature=0.5
    )

# Step 2: Custom Prompt
CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know — don't try to make up an answer.
Only respond based on the context provided.

Context: {context}
Question: {question}

Answer:
"""

def set_custom_prompt(template):
    return PromptTemplate(template=template, input_variables=["context", "question"])

# Step 3: Load Vector DB
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.load_local(
    folder_path=DB_FAISS_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True
)

# Step 4: Create QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(GROQ_MODEL),
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type="stuff",
    chain_type_kwargs={
        'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
    }
)

# Step 5: Query it
user_query = input("Write Query Here: ")
response = qa_chain.invoke({'query': user_query})

print("\n RESULT:\n", response["result"])
print("\n SOURCE DOCUMENTS:")
for doc in response["source_documents"]:
    print("-", doc.metadata.get("source", "[Unknown source]"))
