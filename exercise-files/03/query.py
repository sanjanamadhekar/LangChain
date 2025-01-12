from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
import warnings
import os

warnings.filterwarnings("ignore")
load_dotenv()

# Initialize LLM and Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI()

# Contextualize question prompt
contextualize_q_system_prompt = """Given a chat history and the latest user question {input}, \
formulate a standalone question which can be understood without the chat history. \
Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# QA system prompt
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.

Context: {context}
Question: {input}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Indexing
def initialize_vectorstore(document_path):
    """Initialize the vector store from a given document path"""
    try:
        documents = TextLoader(document_path).load()
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0, separator="\n")
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            collection_name="faq_collection",
            persist_directory="./chroma_db"
        )
        return vectorstore
    except Exception as e:
        print(f"Error initializing vector store: {e}")
        return None

# Initialize vector store (adjust path as needed)
vectorstore = initialize_vectorstore("./docs/faq.txt")

# Retriever setup
def setup_retriever(vectorstore):
    """Set up the retriever with the vector store"""
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

retriever = setup_retriever(vectorstore)

def format_docs(docs):
    """Format retrieved documents into a single string"""
    return "\n\n".join(doc.page_content for doc in docs)

# Chat history management
chat_history = []

# For simple single-turn dialogues, we can use the following enhanced query function 
def query(input_text):
    """
    Main query function that returns a response to the user input
    
    Args:
        input_text (str): User's input question
    
    Returns:
        str: Generated response
    """
    # Check if vectorstore and retriever are initialized
    if vectorstore is None or retriever is None:
        return "Error: Vector store not initialized. Cannot process query."

    try:
        # Retrieve relevant documents
        retrieved_docs = retriever.invoke(input_text)
        context = format_docs(retrieved_docs)

        # Generate response using the context
        response = llm.invoke(
            qa_prompt.format_messages(
                chat_history=chat_history,
                input=input_text,
                context=context
            )
        ).content

        # Update chat history
        chat_history.extend([
            HumanMessage(content=input_text), 
            HumanMessage(content=response)
        ])
        
        return response
    
    except Exception as e:
        # Fallback for any unexpected errors
        print(f"Error processing query: {e}")
        return f"I'm sorry, but I encountered an error: {str(e)}"
    
# For complex multi-turn dialogues, we can use the following enhanced query function   
def query(input_text):
    """
    Enhanced query processing function with context reformulation
    """
    # Validate vector store initialization
    if vectorstore is None or retriever is None:
        return "Error: Vector store not initialized. Cannot process query."

    try:
        # Step 1: Reformulate the question considering chat history
        if chat_history:
            # Use language model to reformulate the question
            reformulated_query = llm.invoke(
                contextualize_q_prompt.format_messages(
                    chat_history=chat_history,
                    input=input_text
                )
            ).content
        else:
            # If no chat history, use original input
            reformulated_query = input_text

        # Step 2: Retrieve relevant documents based on reformulated query
        retrieved_docs = retriever.invoke(reformulated_query)
        context = format_docs(retrieved_docs)

        # Step 3: Generate response using retrieved context
        response = llm.invoke(
            qa_prompt.format_messages(
                chat_history=chat_history,
                input=reformulated_query,
                context=context
            )
        ).content

        # Step 4: Update chat history
        chat_history.extend([
            HumanMessage(content=input_text),
            HumanMessage(content=response)
        ])
        
        return response
    
    except Exception as e:
        print(f"Error processing query: {e}")
        return f"I'm sorry, but I encountered an error: {str(e)}"


# Optional: Add a cleanup method
def cleanup_vectorstore():
    """Clean up the Chroma vector store"""
    if vectorstore:
        try:
            vectorstore.delete_collection()
            print("Vector store collection deleted.")
        except Exception as e:
            print(f"Error deleting vector store collection: {e}")

# Ensure proper cleanup when the script is about to exit
import atexit
atexit.register(cleanup_vectorstore)