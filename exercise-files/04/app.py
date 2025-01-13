import streamlit as st
import openai
import os
from dotenv import load_dotenv
from main import query

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY
MODEL_ENGINE = "gpt-3.5-turbo"

st.title("🤖 Q&A App")
chat_placeholder = st.empty() 


def init_chat_history():
    """Initialize chat history with a system message."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant! Ask me anything"}] 
        

def start_chat():
    """Start the chatbot conversation."""
    # Display chat messages from history on app rerun
    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.markdown(message["content"])
                    
    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # generate response from the user input
        response = query(prompt)
            
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            
        st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
        

if __name__ == "__main__":
    init_chat_history()
    start_chat()