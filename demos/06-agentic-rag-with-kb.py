"""
Demo 06 — Agentic RAG with Azure AI Search Knowledge Base.

Shows how an agent can autonomously retrieve and reason over documents
stored in an Azure AI Search Knowledge Base. The AzureAISearchContextProvider
runs in "agentic" mode, letting the agent plan and execute multi-hop
queries across the knowledge base without any custom retrieval code.

Flow:

 Question ──▶ Agent ─────────────────▶ LLM ──▶ Answer
                │                       ▲
                │  plans & executes     │ retrieved context
                │  search queries       │
                ▼                       │
            ┌────────────┐              │
            │ Azure AI   │──────────────┘
            │ Search     │
            │ Knowledge  │
            │ Base       │
            └────────────┘

Prerequisites:
  - Azure AI Search service with an indexed Knowledge Base
  - Azure OpenAI deployment for chat completions

Environment variables (see .env):
  - AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_CHAT_DEPLOYMENT
  - AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KNOWLEDGE_BASE_NAME
"""

import asyncio
import os
from agent_framework import Agent
from agent_framework.azure import AzureAISearchContextProvider
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv

load_dotenv(override=True)


def print_step(number, title):
    print(f"\n{'=' * 50}")
    print(f"  STEP {number}: {title}")
    print(f"{'=' * 50}\n")


async def main():
    # ── Config ──
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    model = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    search_endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
    knowledge_base_name = os.environ["AZURE_SEARCH_KNOWLEDGE_BASE_NAME"]

    # ── Step 1: Set up clients ──
    print_step(1, "Setting up Azure OpenAI client")
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, "https://cognitiveservices.azure.com/.default"
    )
    client = OpenAIChatClient(
        base_url=f"{endpoint}/openai/v1/",
        api_key=token_provider,
        model_id=model,
    )
    print(f"  Endpoint: {endpoint}")
    print(f"  Model:    {model}")

    # ── Step 2: Set up Azure AI Search context provider ──
    print_step(2, "Setting up Azure AI Search context provider")

    search_credential = DefaultAzureCredential()
    search_provider = AzureAISearchContextProvider(
        endpoint=search_endpoint,
        credential=search_credential,
        knowledge_base_name=knowledge_base_name,
        mode="agentic",
    )
    print(f"  Search endpoint:  {search_endpoint}")
    print(f"  Knowledge Base:   {knowledge_base_name}")
    print(f"  Mode:             agentic")

    # ── Step 3: Create agent with search context ──
    print_step(3, "Creating agent with knowledge base")
    agent = Agent(
        client=client,
        name="search-agent",
        instructions=(
            "You are an AI assistant that helps users learn information from a knowledge base. "
            "Answer questions using the information provided in the context. "
            "Cite your sources when possible. "
            "If no relevant information is found, say you don't have information about that topic."
        ),
        context_providers=[search_provider],
    )
    print(f"  Agent ready with AzureAISearchContextProvider")

    # ── Step 4: Run multi-turn conversation ──
    print_step(4, "Running multi-turn conversation")

    async with search_provider:
        session = agent.create_session()

        # Turn 1
        user_msg = "How can business use generative AI?"
        print(f"  Question: {user_msg}\n")
        response = await agent.run(user_msg, session=session)
        print(f"  Answer: {response.text}\n")

        # Turn 2 — follow-up referencing the previous answer
        print(f"  {'─' * 46}\n")
        user_msg = "What tasks they should start with?"
        print(f"  Follow-up: {user_msg}\n")
        response = await agent.run(user_msg, session=session)
        print(f"  Answer: {response.text}")

        # # Turn 1
        # user_msg = "What does Jason Haley blog about?"
        # print(f"  Question: {user_msg}\n")
        # response = await agent.run(user_msg, session=session)
        # print(f"  Answer: {response.text}\n")

        # # Turn 2 — follow-up referencing the previous answer
        # print(f"  {'─' * 46}\n")
        # user_msg = "What are some of his recent posts about?"
        # print(f"  Follow-up: {user_msg}\n")
        # response = await agent.run(user_msg, session=session)
        # print(f"  Answer: {response.text}")

    await credential.close()
    await search_credential.close()

    print(f"\n{'=' * 50}")
    print("  Done!")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    asyncio.run(main())
