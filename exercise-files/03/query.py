import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import (
    CharacterTextSplitter,
)
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_core.documents import Document
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text_to_embed):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input= [text_to_embed]
    )
    print(response.data[0].embedding)

template: str = """/
    You are a customer support specialist /
    question. 
    You assist users with general inquiries based on /
    and  technical issues. /
    """
    
# define prompt
system_message_prompt_template = SystemMessagePromptTemplate.from_template(template)
chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt_template, HumanMessagePromptTemplate.from_template("{user_query}")])

# init model
model = ChatOpenAI()

# indexing
def load_split_documents():
    """Load a file from path, split it into chunks, embed each chunk and load it into the vector store."""
    raw_text = TextLoader("./docs/faq.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size= 30, chunk_overlap=0, separator=".")
    chunks = text_splitter.split_documents(raw_text)
    # print(f"Number of chunks:, {len(chunks)}")
    # print(chunks[0])
    return chunks

# convert to embeddings
def load_embeddings(documents, user_query):
    """
    Create a vector store from a set of documents and perform a similarity search.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"  
    )
    vector_store.add_texts([doc.page_content for doc in documents])
    docs = vector_store.similarity_search(user_query, k=1)  # Retrieve top 3 results
    print(docs)
    # get_embedding(user_query)
    # _ = [get_embedding(doc.page_content) for doc in docs]


def generate_response(query):
    """Generate a response to a user query."""
    chain = chat_prompt_template | model | StrOutputParser()
    return chain.invoke({"user_query":query})


def query(query_text):
    """Query the model with a user query."""
    documents = load_split_documents()
    load_embeddings(documents, query_text)
    return generate_response(query_text)

query("What is the return policy?")