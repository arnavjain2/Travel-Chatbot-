# Travel Chatbot: Multi-Agent Travel Planning Assistant

## Overview

This project is an AI-powered travel planning assistant that uses a multi-agent workflow to help users plan trips, book flights and hotels, recommend restaurants and more. It leverages advanced LLMs (Large Language Models) and external APIs to provide a seamless, interactive travel planning experience.

The system is built using [LangGraph]for orchestrating agent workflows, [LangChain] for LLM integration, and [Streamlit]for the frontend interface.

---

## Features

- **Conversational travel planning**: Chat with the assistant to plan your trip step by step.
- **Multi-agent architecture**: Specialized agents handle itinerary, flights, hotels, transportation, restaurants and general queries.
- **Memory**: Remembers conversation history for context-aware responses.
- **External API integration**: Uses Amadeus for flights, Tavily for web search and more.
- **Interactive frontend**: Built with Streamlit for a user-friendly chat interface.

---

## Agents

- **Supervisor Agent**: Routes user requests to the appropriate specialized agent based on context.
- **General Agent**: Handles non-travel queries and gently steers users back to travel planning.
- **Itinerary Agent**: Collects trip details (dates, locations, travelers, budget) and creates a high-level itinerary.
- **Flight Agent**: Searches for and recommends flights using the Amadeus API.
- **Hotels Agent**: Suggests hotel options based on destination, dates, and preferences.
- **Transportation Agent**: Recommends local transportation options (public transit, rideshare, rentals).
- **Restaurants Agent**: Provides dining recommendations tailored to user preferences.
- **Validator Agent**: Checks if the workflow is complete and either finishes or routes back for more info.

---

## Technologies & Libraries Used

- **Python 3.11+**
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Streamlit](https://streamlit.io/)
- [Amadeus Python SDK](https://github.com/amadeus4dev/amadeus-python)
- [Tavily Search API](https://docs.tavily.com/)
- [Azure OpenAI](https://azure.microsoft.com/en-us/products/ai-services/openai-service)
- [dotenv](https://pypi.org/project/python-dotenv/)
- [pydantic](https://docs.pydantic.dev/)
- [redis](https://redis.io/) (optional, for advanced memory)
- Other dependencies: `langchain-tavily`, `langchain-openai`, `langchain-groq`, `langchain-experimental`

---

## Setup Instructions

### 1. Clone the Repository

```sh
git clone <your-repo-url>
cd travel_chatbot
```

### 2. Create and Activate a Python Environment

It is recommended to use [conda](https://docs.conda.io/en/latest/) (as per `.vscode/settings.json`):

```sh
conda create -n travel_env python=3.11
conda activate travel_env
```

Or use `venv`:

```sh
python3 -m venv travel_env
source travel_env/bin/activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

If `requirements.txt` is not present, install manually:

```sh
pip install langchain langchain_groq langchain_community langgraph rizaio langchain_tavily langchain-openai amadeus streamlit python-dotenv pydantic redis
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root (already present in this repo). Fill in your API keys:

```
GOOGLE_API_KEY=...
AMADEUS_CLIENT_ID=...
AMADEUS_CLIENT_SECRET=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_DEPLOYMENT_NAME=...
AZURE_OPENAI_API_VERSION=...
TAVILY_API_KEY=...
GROQ_API_KEY=...
SERP_API_KEY=...
OPENAI_API_KEY=...
```

**Never share your `.env` file publicly.**

### 5. Run the Backend (for testing)

```sh
python Agents.py
```

This will execute the agent workflow and print outputs to the terminal.

### 6. Run the Frontend (Streamlit App)

```sh
streamlit run frontend.py
```

Open the provided URL in your browser to interact with the chatbot.

---

## Project Structure

```
Agents.py           # Main multi-agent workflow and backend logic
frontend.py         # Streamlit frontend for chat interface
.env                # Environment variables (API keys, secrets)
workflow_graph.png  # Visualization of the agent workflow
.vscode/            # VS Code settings
.gitignore
```

---

## How It Works

1. **User Input**: The user sends a message (e.g., "Plan a trip to Paris").
2. **Supervisor Agent**: Determines which specialized agent should handle the request.
3. **Specialized Agent**: Collects information or performs actions (e.g., finds flights, hotels).
4. **Validator Agent**: Checks if the user's request is fully satisfied.
5. **Workflow**: Continues until the travel plan is complete or the user is satisfied.

The system uses memory to maintain context across turns, allowing for a natural, conversational experience.

---

## Example Usage

- "Can you help me plan a trip to Tokyo next month?"
- "Find me flights from NYC to Paris for two people."
- "Suggest some vegetarian restaurants in Rome."
- "What did I say earlier?"

---

## Troubleshooting

- Ensure all API keys in `.env` are correct and active.
- If you see errors about missing packages, re-run the `pip install` command.
- For issues with Amadeus or Tavily, check their API documentation and usage limits.

---

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Streamlit](https://streamlit.io/)
- [Amadeus for Developers](https://developers.amadeus.com/)
- [Tavily Search](https://tavily.com/)

---