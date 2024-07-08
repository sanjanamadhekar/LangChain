from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from typing import List, Optional
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)


# Prompt with Query analysis

# Create Index & Connect to datasource
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = ["Harrison worked at Kensho"]
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="harrison")
retriever_harrison = vectorstore.as_retriever(search_kwargs={"k": 1})

texts = ["Ankush worked at Facebook"]
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="ankush")
retriever_ankush = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = vectorstore.similarity_search("who worked at Facebook?")
print(docs[0].page_content)


# Retrieval with Query analysis