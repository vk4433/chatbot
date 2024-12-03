import os
from dotenv import load_dotenv
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
google_api=os.getenv("google_api")


st.title("Lawroom ai Task")

def load():
    #Legislative Department
    loader = PyPDFDirectoryLoader("data")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(data)

f_doc = load()
def vec_store(data):
    embeddings = GoogleGenerativeAIEmbeddings(model ="models/embedding-001",google_api_key=google_api)
    return FAISS.from_documents(data,embeddings)
vector = vec_store(f_doc)
   
def search_query(query):
    return vector.similarity_search(query)

def gemini(content,query):
    system_message = SystemMessage(content="You are a helpful assistant. Provide  answers based on the given context and query, only if it relates to law.")
    human_message = HumanMessage(content=f"Context:\n{context}\n\nQuestion:\n{query}")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",api_key=google_api)
    chat =llm.invoke([system_message,human_message])
    return chat.content




query = st.text_input("please enter your question")


if query:
        results = search_query(query)
        context = "\n\n".join([result.page_content for result in results])
        st.subheader("Retrieved Context:")
        st.write(context)

        answer = gemini(context, query)
    
        st.subheader("using api Answer:")
        st.write(answer)






