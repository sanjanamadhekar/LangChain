from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(model="gpt-4-turbo-preview")

# Connect to your Atlas cluster

# Define collection and index name

# Load the sample data (PDF document)

# Split PDF into documents

# Print the first document

# Instantiate the vector store

def query_data(query):
    """run vector search queries"""
    results = vector_search.similarity_search(query)
    print(results)

query_data("MongoDB Atlas Sec")

