from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.agents import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from retrievers.vector_search_index import rag_chain
from retrievers.web_search import search_engine_chain

load_dotenv()

# load the LLM

# define the tool

# create the prompt

# bind the tool to the LLM

# create the agent

# run the agent
