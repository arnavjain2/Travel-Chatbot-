import os
import json
import pprint
import redis
from dotenv import load_dotenv
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
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver() # MemorySaver is a built-in memory storage system provided by LangGraph 
# which stores history in RAM (in memory only — not in a file or database unless we use a different checkpointer).


# Load environment variables from .env file
load_dotenv()

# Access environment variables safely
groq_api_key = os.getenv("GROQ_API_KEY")
riza_api_key = os.getenv("RIZA_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")

amadeus_client_id = os.getenv("AMADEUS_CLIENT_ID")
amadeus_client_secret = os.getenv("AMADEUS_CLIENT_SECRET")

google_api_key = os.getenv("GOOGLE_API_KEY")
serp_api_key = os.getenv("SERP_API_KEY")

azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

rapidapi_key = os.getenv("RAPIDAPI_KEY")

# Setup Azure LLM client
llm_azure = AzureChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    api_version=azure_openai_api_version  # Use the env variable here
)

# # Invoke example
# response_azure = llm_azure.invoke("Give me a packing list for a beach vacation in Spain.")
# print("Azure GPT-4:", response_azure.content)

# Initialize tools
tool_tavily = TavilySearch(max_results=2)
tool_code_interpreter = PythonREPLTool()

# Supervisor Agent 
# # Dummy Command, HumanMessage, and MessagesState classes for testing
# class Command:
#     def __init__(self, update, goto):
#         self.update = update
#         self.goto = goto

# class HumanMessage:
#     def __init__(self, content, name="user"):
#         self.role = "user"
#         self.content = content
#         self.name = name

# MessagesState = dict  # Use your framework's type if available

# System prompt for Travel Planning Supervisor
system_prompt = ('''
You are a travel planning supervisor managing a team of 5 agents: Itinerary, Flight, Hotels, Transportation, and Restaurants.
Your role is to guide the planning process based on the user's travel request, asking clarifying questions if needed, and routing tasks to the appropriate agent.

**Team Members**:
1. **General**: Used when the user input is not related to travel planning. This agent answers general queries and then offers to help with travel if needed.
2. **Itinerary**: First point of contact. Responsible for collecting key travel details: travel dates, origin, destination, number of travelers, budget, and preferred currency. It then outlines a high-level itinerary structure.
3. **Flight**: Handles search and recommendation of flights, based on origin, destination, and travel dates.
4. **Hotels**: Suggests accommodation options according to the itinerary, budget, and preferences.
5. **Transportation**: Recommends local transportation options (e.g., rental cars, metro passes, airport transfers).
6. **Restaurants**: Suggests dining options tailored to traveler preferences, dietary needs, and locations in the itinerary.

**Supervisor Responsibilities**:
1. Review user input and assess whether key travel details are provided.
2. If the request is vague or missing travel information, route it to **Itinerary** first for clarification.
3. If the user input is unrelated to travel planning, route it to **General**.
4. Once the necessary information is collected, route to the relevant agents to complete the planning process.
5. Ensure the workflow progresses logically, until all travel needs are addressed.

Your goal is to create a complete and enjoyable travel plan by efficiently coordinating between agents, asking for more information when necessary, and ensuring each stage is addressed.
''')

# Define Supervisor schema using Pydantic
class Supervisor(BaseModel):
    next: Literal["general", "itinerary", "flight", "hotels", "transportation", "restaurants"] = Field(
        description="Specifies the next worker in the pipeline: "
                    "'general' for general queries."
                    "'itinerary' for clarifying or gathering travel details, "
                    "'flight' for searching flights, "
                    "'hotels' for booking accommodations, "
                    "'transportation' for local travel, "
                    "'restaurants' for dining recommendations."
    )
    reason: str = Field(
        description="The reason for the decision, providing context on why a particular worker was chosen."
    )

    @staticmethod
    def supervisor_node(state: MessagesState) -> Command[Literal["general", "itinerary", "flight", "hotels", "transportation", "restaurants"]]:
        messages = [
            {"role": "system", "content": system_prompt},
        ] + state["messages"]

        response = llm_azure.with_structured_output(Supervisor).invoke(messages)

        goto = response.next
        reason = response.reason

        print(f"Current Node: Supervisor -> Goto: {goto}")
        print(f"Reason: {reason}")

        return Command(
            update={
                "messages": [
                    HumanMessage(content=reason, name="supervisor")
                ]
            },
            goto=goto,
        )
        
# # DummyLLM for local testing
# class DummyLLM:
#     def with_structured_output(self, schema):
#         class Invoker:
#             def invoke(self_inner, messages):
#                 # Simulated logic: always route to itinerary for clarification
#                 return schema(
#                     next="itinerary",
#                     reason="Initial request lacks travel details. Routing to Itinerary agent for clarification."
#                 )
#         return Invoker()

# # Instantiate DummyLLM for testing
# llm = DummyLLM()

# # Simulate user input
# example_state = {
#     "messages": [
#         {"role": "user", "content": "Can you help me plan a trip?"}
#     ]
# }

# # Run supervisor node
# result = Supervisor.supervisor_node(example_state)
# print("Command returned:", result.goto)
# print("Supervisor message:", result.update["messages"][0].content)

# General Agent 
def general_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Handles general, non-travel queries using the LLM, and then invites the user
    to start planning a trip.
    """

    # System prompt to guide the general-purpose agent
    system_prompt = (
        "You are a friendly assistant who handles general (non-travel-specific) user queries.\n"
        "- If the query is unrelated to travel (like programming, trivia, or casual questions), answer briefly and helpfully.\n"
        "- Then, politely steer the user back toward travel planning by offering help with itineraries, destinations, or flights.\n\n"
        "Be conversational, respectful, and enthusiastic about travel!"
    )

    # Create a ReAct agent for restaurant search
    general_agent = create_react_agent(
        llm_azure,
        tools=[tool_tavily]
    )

    # Invoke the agent to process the request and return results
    result = general_agent.invoke(state)

    print("Current Node: General Agent -> Goto: Supervisor")

    # Return a command to add itinerary output to the conversation and return to supervisor
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content,
                    name="general"
                )
            ]
        },
        goto="validator"
    )

# Itinerary Agent  
def itinerary_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Itinerary agent node to collect and organize essential travel information.

    Args:
        state (MessagesState): The current conversation state.
    Returns:
        Command: A command to update the state and return to the supervisor.
    """

    # System prompt tailored for the Itinerary Agent
    system_prompt = (
        "You are an expert travel planner. Your job is to create or refine a high-level travel itinerary "
        "based on the user's input. Start by checking if the following details are provided:\n"
        "1. Departure location\n"
        "2. Destination\n"
        "3. Dates of travel\n"
        "4. Number of travelers\n"
        "5. Budget and preferred currency\n\n"
        "If any of these details are missing, ask follow-up questions to get them. "
        "Be conversational and helpful. Once you have all the details, return a short summary itinerary "
        "outlining what the user wants to do, including travel goals if mentioned.\n"
    )

    # Create a ReAct agent for restaurant search
    itinerary_agent = create_react_agent(
        llm_azure,
        tools=[tool_tavily]
    )

    # Invoke the agent to process the request and return results
    result = itinerary_agent.invoke(state)

    print("Current Node: Restaurants Agent -> Goto: Supervisor")

    # Return a command to add itinerary output to the conversation and return to supervisor
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content,
                    name="itinerary"
                )
            ]
        },
        goto="validator",
    )
    
# Flight Agent
# Amadeus client setup 
amadeus = Client(
    client_id=os.environ["AMADEUS_CLIENT_ID"],
    client_secret=os.environ["AMADEUS_CLIENT_SECRET"]
)

# Define Amadeus wrapper function
def search_flights_amadeus(query: str) -> str:
    """
    Query format expected: JSON string with keys:
    - origin
    - destination
    - departure_date (YYYY-MM-DD)
    - return_date (YYYY-MM-DD)
    - adults (int)
    """
    try:
        params = json.loads(query)
        response = amadeus.shopping.flight_offers_search.get(
            originLocationCode=params["origin"],
            destinationLocationCode=params["destination"],
            departureDate=params["departure_date"],
            returnDate=params["return_date"],
            adults=params.get("adults", 1)
        )
        offers = response.data[:3]
        results = []
        for offer in offers:
            itinerary = offer['itineraries'][0]['segments'][0]
            result = {
                "airline": itinerary['carrierCode'],
                "departure": itinerary['departure']['at'],
                "arrival": itinerary['arrival']['at'],
                "duration": itinerary['duration'],
                "price": offer['price']['total'] + " " + offer['price']['currency']
            }
            results.append(result)
        return json.dumps(results, indent=2)
    except ResponseError as e:
        return f"Amadeus API error: {e}"
    except Exception as e:
        return f"Tool input error: {e}"
    
tool_amadeus_flights = Tool(
    name="amadeus_flight_search",
    func=search_flights_amadeus,
    description=(
        "Use this to find flight options. Input must be a JSON string with keys: "
        "`origin`, `destination`, `departure_date`, `return_date`, and `adults`."
    )
)

def flight_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Flight agent node for searching and recommending flights based on user's travel details.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with flight search results and route to the supervisor.
    """

    # System prompt to guide the ReAct flight agent
    system_prompt = (
        "You are a helpful flight booking assistant. Your job is to find the best flight options for the user's trip "
        "using provided travel details. Assume the user has already given:\n"
        "1. Departure location\n"
        "2. Destination\n"
        "3. Travel dates (departure and return)\n"
        "4. Number of travelers\n"
        "\nUse external tools if available to look up realistic and recent flight options.\n"
        "Summarize results with: airline, flight times, duration, and approximate price.\n"
        "Be concise, and provide 2–3 good options when possible."
    )

    # Create a ReAct agent for flight search
    flight_agent = create_react_agent(
        llm_azure,
        tools=[tool_amadeus_flights]
    )

    # Use the flight agent to process the query
    result = flight_agent.invoke(state)

    print("Current Node: Flight Agent -> Goto: Supervisor")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=result["messages"][-1].content,
                    name="flight"
                )
            ]
        },
        goto="validator",
    )

# Hotels Agent
def hotels_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Hotels agent node for recommending hotel options based on user's travel details.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with hotel recommendations and route to the supervisor.
    """

    # System prompt to guide hotel search logic
    system_prompt = (
        "You are a helpful hotel booking assistant. Based on the user’s itinerary, your task is to recommend suitable "
        "hotels in their destination city.\n\n"
        "Assume the user has already provided:\n"
        "1. Destination city\n"
        "2. Travel dates\n"
        "3. Number of travelers\n"
        "4. Budget and currency (if available)\n\n"
        "Use web tools if needed to suggest 2–3 hotel options. Each recommendation should include:\n"
        "- Hotel name\n"
        "- Location (neighborhood or distance to city center/landmarks)\n"
        "- Price per night\n"
        "- Rating or review summary (if available)\n\n"
        "Tailor suggestions to the user's stated preferences and budget if provided."
    )

    # Create a ReAct agent for hotel lookup
    hotels_agent = create_react_agent(
        llm_azure,
        tools=[tool_tavily]  # Replace with real hotel search tool or web lookup agent
    )

    # Invoke the agent with the current conversation state
    result = hotels_agent.invoke(state)

    print("Current Node: Hotels Agent -> Goto: Supervisor")

    # Return a command with the generated message and next node
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="hotels")
            ]
        },
        goto="validator",
    )
    
# Transportation Agent
def transportation_node(state: MessagesState) -> Command[Literal["validator"]]:

    """
    Transportation agent node for recommending local travel options based on user's itinerary.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with transportation suggestions and route to the supervisor.
    """

    # System prompt to guide the transportation planning
    system_prompt = (
        "You are a travel assistant that helps users plan local transportation for their trip.\n\n"
        "Based on the user's destination and itinerary, recommend how they can move around locally.\n"
        "Suggestions may include:\n"
        "- Airport transfers (e.g., taxi, shuttle, train)\n"
        "- Public transportation options (subways, buses, metro cards)\n"
        "- Ride-sharing apps (Uber, Lyft, etc.)\n"
        "- Rental cars or scooters\n\n"
        "Take into account:\n"
        "- Destination city\n"
        "- Trip duration\n"
        "- Group size\n"
        "- Budget or preferences (if provided)\n\n"
        "Return 2–3 reasonable transportation suggestions with brief descriptions and approximate pricing (if available)."
    )

    # Create a ReAct agent for transportation recommendations
    transportation_agent = create_react_agent(
        llm_azure,
        tools=[tool_tavily]  # Or any local transportation API/search tool
    )

    # Invoke the agent to generate transportation options
    result = transportation_agent.invoke(state)

    print("Current Node: Transportation Agent -> Goto: Supervisor")

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="transportation")
            ]
        },
        goto="validator",
    )

# Restaurants Agent
def restaurants_node(state: MessagesState) -> Command[Literal["validator"]]:
    """
    Restaurants agent node for recommending dining options based on the user's travel plans and preferences.

    Args:
        state (MessagesState): The current state containing the conversation history.

    Returns:
        Command: A command to update the state with restaurant suggestions and route to the supervisor.
    """

    # System prompt for restaurant recommendation logic
    system_prompt = (
        "You are a knowledgeable restaurant guide for travelers. Based on the user’s itinerary and preferences, "
        "suggest great places to eat during their trip.\n\n"
        "Use the following information to guide your response:\n"
        "- Destination city or neighborhood\n"
        "- Trip duration or specific days\n"
        "- Cuisine preferences (if mentioned)\n"
        "- Dietary restrictions (e.g., vegetarian, gluten-free)\n"
        "- Budget (if stated)\n\n"
        "Return 2–3 restaurant recommendations with:\n"
        "- Restaurant name\n"
        "- Cuisine type\n"
        "- Neighborhood or distance from key locations\n"
        "- Price range and review rating (if available)\n"
        "Tailor recommendations to be friendly, specific, and helpful."
    )

    # Create a ReAct agent for restaurant search
    restaurants_agent = create_react_agent(
        llm_azure,
        tools=[tool_tavily]
    )

    # Invoke the agent to process the request and return results
    result = restaurants_agent.invoke(state)

    print("Current Node: Restaurants Agent -> Goto: Supervisor")

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="restaurants")
            ]
        },
        goto="validator",
    )

# Validator Agent
system_prompt = '''
You are a travel workflow validator. Your task is to determine whether the workflow has been completed successfully,
based on whether the user's original query has been clearly and appropriately handled.

Instructions:
1. Carefully read the user's initial request (first message).
2. Evaluate the final response (last message).
3. Determine whether the response is complete and appropriate **for the type of query**.

Evaluation Criteria:

**For travel-related queries**, ensure the final answer includes:
    - Itinerary summary
    - Flight suggestions (if needed)
    - Hotel recommendations
    - Local transportation advice
    - Restaurant suggestions (optional but preferred)
    - All major trip details the user asked for

**For general or non-travel-related queries**, ensure:
    - The user’s request was acknowledged or answered correctly and clearly
    - The assistant also offered to help plan a trip or redirected the user toward travel planning

Routing Rules:
- If the final response fully satisfies the request (travel or general), respond with: FINISH
- If the final response is incomplete, vague, or off-topic, respond with: supervisor

Your output should include:
- 'next': Either 'FINISH' or 'supervisor'
- 'reason': A brief explanation of your decision
'''

class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(
        description="Route to 'supervisor' to continue workflow, or 'FINISH' to terminate if complete."
    )
    reason: str = Field(
        description="Explanation of the decision, assessing completeness and alignment with the user query."
    )

def validator_node(state: MessagesState) -> Command[Literal["supervisor", END]]:
    """
    Validator node for evaluating whether the travel planning workflow has produced a complete response.

    Args:
        state (dict): The conversation history and state.

    Returns:
        Command: Indicates whether to end the workflow or send it back to the supervisor.
    """

    user_question = state["messages"][0].get("content", "")
    agent_answer = state["messages"][-1].get("content", "")

    # Prepare messages for the LLM
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": agent_answer},
    ]

    # Invoke the LLM with structured output using Validator schema
    response = llm_azure.with_structured_output(Validator).invoke(messages)

    goto = response.next
    reason = response.reason

    # Map 'FINISH' to END constant (replace END with your actual end token)
    if goto == "FINISH":
        goto = END
        print("Validator Decision: Workflow complete -> Transitioning to END")
    else:
        print("Validator Decision: Workflow incomplete -> Routing to Supervisor")

    return Command(
        update={
            "messages": [
                HumanMessage(content=reason, name="validator")
            ]
        },
        goto=goto,
    )
    
# Define validator_node and Validator class
class Validator(BaseModel):
    next: Literal["supervisor", "FINISH"] = Field(...)
    reason: str = Field(...)

def validator_node(state: MessagesState) -> Command[Literal["supervisor", END]]:
    ...

# Initialize the StateGraph with MessagesState
builder = StateGraph(MessagesState)

# Add edges and nodes to define the workflow of the graph
builder.add_edge(START, "supervisor")  # Connect the start node to the supervisor node

# Add all nodes representing agents/workflow steps
builder.add_node("supervisor", Supervisor.supervisor_node)
builder.add_node("general", general_node)
builder.add_node("itinerary", itinerary_node)
builder.add_node("flight", flight_node)
builder.add_node("hotels", hotels_node)
builder.add_node("transportation", transportation_node)
builder.add_node("restaurants", restaurants_node)
builder.add_node("validator", validator_node)

# Compile the graph to finalize its structure
graph = builder.compile(checkpointer=memory)

# Generate and save PNG visualization
png_bytes = graph.get_graph().draw_mermaid_png()

# Save to file (for VS Code)
with open("workflow_graph.png", "wb") as f:
    f.write(png_bytes)
    
# Define input messages 
inputs = {
    "messages": [
        HumanMessage(content="Hi")
    ]
}

# Config with required thread_id
config = {"configurable": {"thread_id": "1"}}

# Example usage of the graph with a sample input
for output in graph.stream(inputs, config=config):
    for key, value in output.items():
        if value is None:
            continue
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint(value, indent=2, width=80, depth=None)
        print(value['messages'][-1].content)  # Access last item of the messages list

# Example continued to check memory functionality
inputs = {
    "messages": [
        HumanMessage(content="What did I say?")
    ]
}

config = {"configurable": {"thread_id": "1"}}

for output in graph.stream(inputs, config=config): 
    for key, value in output.items():
        if value is None:
            continue
        pprint.pprint(f"Output from node '{key}':")
        pprint.pprint(value, indent=2, width=80, depth=None)
        
        if "messages" in value and value["messages"]:
            print(value["messages"][-1].content)
            

# # Example usage of the graph with a sample input
# inputs = {
#     "messages": [
#         HumanMessage(content="Hi")

#     ]
# }

# for output in graph.stream(inputs):
#     for key, value in output.items():
#         if value is None:
#             continue
#         pprint.pprint(f"Output from node '{key}':")
#         pprint.pprint(value, indent=2, width=80, depth=None)
#         print()





