import streamlit as st
import os
import json
import pprint
import redis
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from Agents import graph, MessagesState
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_groq import ChatGroq
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from amadeus import Client, ResponseError
from typing import Literal 
from typing import Annotated, Sequence, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, MessagesState, START , END

# Load environment variables
load_dotenv()

# Page configuration 
st.set_page_config(
    page_title="Travel Planning Assistant",
    page_icon="✈️",
    layout="wide"
)

# Initialize session state
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "1"  # default thread ID for memory

# Title and description
st.title("Travel Planning Assistant")
st.markdown("Hi! I am your personal travel assistant. Ask me anything about travel planning, flights, hotels, restaurants, and more.")

# Sidebar 
with st.sidebar:
    st.header("How it works")
    st.markdown("""
    Our system uses specialized agents:

    - **General Agent**: Handles non-travel queries  
    - **Itinerary Agent**: Collects travel details and creates itineraries  
    - **Flight Agent**: Searches and recommends flights  
    - **Hotels Agent**: Suggests accommodation options  
    - **Transportation Agent**: Recommends local transport  
    - **Restaurants Agent**: Provides dining recommendations
    """)
    if st.button("🧹 Clear Conversation"):
        # Clear both UI history and LangGraph memory via a new thread ID
        st.session_state.conversation_history = []
        st.session_state.thread_id = str(int(st.session_state.thread_id) + 1)
        st.rerun()

# Chat history UI 
st.subheader("Chat with Travel Assistant")

for message in st.session_state.conversation_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input handling 
if prompt := st.chat_input("Ask me about travel planning or anything else..."):

    # Add user message
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process through graph
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):

            inputs = {"messages": [HumanMessage(content=prompt)]}
            config = {"configurable": {"thread_id": st.session_state.thread_id}}

            try:
                response_parts = []
                status_container = st.empty()
                response_container = st.empty()

                for output in graph.stream(inputs, config=config):
                    for key, value in output.items():
                        if value is None:
                            continue

                        if key != "_end_":
                            current_agent = key.title()
                            status_container.info(f"Current agent: {current_agent}")

                        if "messages" in value and value["messages"]:
                            latest_message = value["messages"][-1]
                            if hasattr(latest_message, "content"):
                                response_parts.append(latest_message.content)
                                full_response = "\n\n".join(response_parts)
                                response_container.markdown(full_response)

                status_container.empty()

                final_response = "\n\n".join(response_parts) if response_parts else "Sorry, no response generated."

                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": final_response
                })

            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": error_message
                })

# Footer 
st.markdown("---")
st.markdown("🛠 Powered by LangGraph Multi-Agent System")

# OPTIONAL: Example Memory Check at Bottom
# (Uncomment to test memory recall)
# """
# example_input = {
#     "messages": [
#         HumanMessage(content="Hey what is my name?")
#     ]
# }
# example_config = {"configurable": {"thread_id": st.session_state.thread_id}}

# for output in graph.stream(example_input, config=example_config):
#     for key, value in output.items():
#         if value is None:
#             continue
#         pprint.pprint(f"Output from node '{key}':")
#         pprint.pprint(value, indent=2)
#         if "messages" in value and value["messages"]:
#             print(value["messages"][-1].content)
# """