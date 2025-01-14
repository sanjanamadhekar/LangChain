from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain import hub
from langchain_core.runnables import chain
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(temperature=0)
llm_function_calling = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Prompt with Query analysis
class Search(BaseModel):
    """Search for information about a person."""

    query: str = Field(
        ...,
        description="Query to look up",
    )
    person: str = Field(
        ...,
        description="Person to look things up for. Should be `HARRISON` or `ANKUSH`.",
    )

system = """You have the ability to issue search queries to get information to help answer user information."""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
structured_llm = llm_function_calling.with_structured_output(Search, method="function_calling")
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm

# structured_output = query_analyzer.invoke("where did Harrison Work")
# print(structured_output)

# Create Index & Connect to datasource
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

texts = ["Harrison worked at Kensho"]
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="harrison", persist_directory="./chroma_db_harrison")
retriever_harrison = vectorstore.as_retriever(search_kwargs={"k": 1})

texts = ["Ankush worked at Facebook"]
vectorstore = Chroma.from_texts(texts, embeddings, collection_name="ankush", persist_directory="./chroma_db_ankush")
retriever_ankush = vectorstore.as_retriever(search_kwargs={"k": 1})

# docs = vectorstore.similarity_search("who worked at Facebook?")
# print(docs[0].page_content)


# Retrieval with Query analysis

retrievers = {
    "HARRISON": retriever_harrison,
    "ANKUSH": retriever_ankush,
}

@chain
def custom_chain(question):
    structured_output = query_analyzer.invoke(question)
    retriever = retrievers[structured_output.person]
    return retriever.invoke(structured_output.query)

response = custom_chain.invoke("where did Ankush work?")
print(response[0].page_content)