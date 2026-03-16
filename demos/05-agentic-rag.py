import asyncio
import os
import dotenv
from typing import Annotated

from openai import AzureOpenAI
from agent_framework import tool
from agent_framework.azure import AzureAIProjectAgentProvider
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.identity.aio import AzureCliCredential
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

import logging
logging.getLogger("agent_framework_azure_ai._client").setLevel(logging.ERROR)

dotenv.load_dotenv(override=True)

SYSTEM_PROMPT = """
You are an AI assistant that helps users learn information from the IT support ticket knowledge base.
You have access to three tools:
1. search_knowledge_base - searches the AI/GenAI knowledge base for articles and reports.
2. search_support_tickets - searches IT support tickets for past issues, resolutions, and status.
3. search_web - searches the web using Bing for current information not found in the knowledge base.

When the user asks a question, choose the appropriate tool(s) to find relevant information.
Prefer the knowledge base and support ticket tools first for internal questions.
Use the web search tool for current events, external information, or when internal sources lack relevant results.
You may call the tools multiple times with different queries if needed.
Answer using only the information returned by the tools.
Cite your sources using the format [source-id].
If the tools return no relevant results, respond with "I don't know."
"""


def print_step(number, title):
    print(f"\n{'=' * 50}")
    print(f"  STEP {number}: {title}")
    print(f"{'=' * 50}\n")


def create_search_tool(
    search_client: SearchClient,
    openai_client: AzureOpenAI,
    embedding_model: str,
    use_hybrid: bool = True,
    top_k: int = 3,
    show_search_results: bool = True,
):
    """Create a search tool closure that captures the clients and config."""

    @tool
    def search_knowledge_base(
        query: Annotated[str, "The search query to find relevant knowledge base articles"],
    ) -> str:
        """Search the knowledge base for information relevant to the query."""
        print(f"\n  🔍 Searching knowledge base for query: {query}")
        # Embed the query
        embed_response = openai_client.embeddings.create(
            model=embedding_model, input=[query],
        )
        q_vector = embed_response.data[0].embedding

        vq = VectorizedQuery(
            vector=q_vector,
            fields="contentVector",
        )

        if use_hybrid:
            results = search_client.search(
                search_text=query,
                vector_queries=[vq],
                top=top_k,
            )
        else:
            results = search_client.search(
                search_text=None,
                vector_queries=[vq],
                top=top_k,
            )

        chunks = []
        for doc in results:
            chunks.append({
                "id": doc["id"],
                "content": doc["content"],
                "score": doc.get("@search.score"),
            })

        print(f"\n  🔍 Tool called: search_knowledge_base")
        print(f"     Query:   {query}")
        print(f"     Results: {len(chunks)} chunks")

        if show_search_results:
            for rank, chunk in enumerate(chunks, 1):
                score = chunk["score"]
                if score is not None:
                    print(f"\n     Result {rank}  (id={chunk['id']}, score={score:.4f})")
                else:
                    print(f"\n     Result {rank}  (id={chunk['id']})")
                print(f"     {chunk['content']}")
                print(f"     {'─' * 40}")

        # Format as context for the LLM
        parts = []
        for chunk in chunks:
            parts.append(f"[source-{chunk['id']}]\n{chunk['content']}")
        return "\n\n".join(parts)

    return search_knowledge_base


def create_support_ticket_tool(
    search_client: SearchClient,
    openai_client: AzureOpenAI,
    embedding_model: str,
    use_hybrid: bool = True,
    top_k: int = 5,
    show_search_results: bool = True,
):
    """Create a support ticket search tool closure."""

    @tool
    def search_support_tickets(
        query: Annotated[str, "The search query to find relevant support tickets"],
    ) -> str:
        """Search IT support tickets for past issues, resolutions, and status."""
        print(f"\n  🔍 Searching support tickets for query: {query}")
        embed_response = openai_client.embeddings.create(
            model=embedding_model, input=[query],
        )
        q_vector = embed_response.data[0].embedding

        # Search across both Body and Answer embeddings
        body_vq = VectorizedQuery(
            vector=q_vector,
            fields="BodyEmbeddings",
        )
        answer_vq = VectorizedQuery(
            vector=q_vector,
            fields="AnswerEmbeddings",
        )

        if use_hybrid:
            results = search_client.search(
                search_text=query,
                vector_queries=[body_vq, answer_vq],
                top=top_k,
            )
        else:
            results = search_client.search(
                search_text=None,
                vector_queries=[body_vq, answer_vq],
                top=top_k,
            )

        tickets = []
        for doc in results:
            tickets.append({
                "id": doc["Id"],
                "subject": doc.get("Subject", ""),
                "body": doc.get("Body", ""),
                "answer": doc.get("Answer", ""),
                "priority": doc.get("Priority", ""),
                "type": doc.get("Type", ""),
                "queue": doc.get("Queue", ""),
                "tags": doc.get("Tags", []),
                "score": doc.get("@search.score"),
            })

        print(f"\n  🎫 Tool called: search_support_tickets")
        print(f"     Query:   {query}")
        print(f"     Results: {len(tickets)} tickets")

        if show_search_results:
            for rank, ticket in enumerate(tickets, 1):
                score = ticket["score"]
                if score is not None:
                    print(f"\n     Result {rank}  (id={ticket['id']}, score={score:.4f})")
                else:
                    print(f"\n     Result {rank}  (id={ticket['id']})")
                print(f"     Subject:  {ticket['subject']}")
                print(f"     Priority: {ticket['priority']}  Type: {ticket['type']}")
                print(f"     Body:     {ticket['body'][:200]}...")
                print(f"     {'─' * 40}")

        # Format as context for the LLM
        parts = []
        for ticket in tickets:
            tags = ", ".join(ticket["tags"]) if ticket["tags"] else "none"
            parts.append(
                f"[ticket-{ticket['id']}]\n"
                f"Subject: {ticket['subject']}\n"
                f"Priority: {ticket['priority']} | Type: {ticket['type']} | Queue: {ticket['queue']}\n"
                f"Tags: {tags}\n"
                f"Issue: {ticket['body']}\n"
                f"Resolution: {ticket['answer']}"
            )
        return "\n\n".join(parts)

    return search_support_tickets


async def main():
    # ── Config ──
    randomizer = "jh"
    index_name = f"{randomizer.lower()}vectorindex"
    support_ticket_index = "support-tickets-index"
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    chat_model = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_API_KEY")
    endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

    #user_query = "Can you explain why only 5% of custom enterprise AI projects succeed?"
    user_query = "What are some issues with Surface devices and how were they resolved?"
    #user_query = "What is the weather in Boston today?"
    #user_query = "What is today's top headline in the news?"
    use_hybrid = True
    show_search_results = False  # Set to False to hide tool call details

    # ── Step 1: Set up clients ──
    print_step(1, "Setting up clients")

    # Sync OpenAI client for embeddings/search tool
    openai_client = AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=get_bearer_token_provider(
            DefaultAzureCredential(),
            "https://cognitiveservices.azure.com/.default",
        ),
        api_version="2024-02-01",
    )

    kb_search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key),
    )
    ticket_search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=support_ticket_index,
        credential=AzureKeyCredential(search_key),
    )

    # Get the full connection resource ID for Bing grounding
    from azure.ai.projects import AIProjectClient
    project_client = AIProjectClient(
        endpoint=os.environ["AZURE_AI_PROJECT"],
        credential=DefaultAzureCredential(),
    )
    bing_connection_name = os.getenv("BING_PROJECT_CONNECTION_ID")
    bing_connection = project_client.connections.get(name=bing_connection_name)
    bing_connection_id = bing_connection.id

    print(f"  Endpoint: {endpoint}")
    print(f"  Model:    {chat_model}")
    print(f"  Indexes:  {index_name}, {support_ticket_index}")
    print(f"  Bing:     {bing_connection_id}")

    # ── Step 2: Create search tools ──
    print_step(2, "Creating search tools")
    kb_tool = create_search_tool(
        kb_search_client, openai_client, embedding_model,
        use_hybrid=use_hybrid,
        show_search_results=show_search_results,
    )
    ticket_tool = create_support_ticket_tool(
        ticket_search_client, openai_client, embedding_model,
        use_hybrid=use_hybrid,
        show_search_results=show_search_results,
    )
    bing_tool = {
        "type": "bing_grounding",
        "bing_grounding": {
            "search_configurations": [
                {
                    "project_connection_id": bing_connection_id,
                }
            ]
        },
    }
    print(f"  Tool 1:  search_knowledge_base  ({index_name})")
    print(f"  Tool 2:  search_support_tickets ({support_ticket_index})")
    print(f"  Tool 3:  bing_grounding         (Bing)")
    print(f"  Mode:    {'Hybrid' if use_hybrid else 'Vector-only'}")

    # ── Step 3: Create agent with tools ──
    print_step(3, "Creating agent")
    async with (
        AzureCliCredential() as credential,
        AzureAIProjectAgentProvider(credential=credential, project_endpoint=os.environ["AZURE_AI_PROJECT"]) as provider,
    ):
        agent = await provider.create_agent(
            name="AgenticRAGAgent",
            model=chat_model,
            instructions=SYSTEM_PROMPT,
            tools=[kb_tool, ticket_tool, bing_tool],
        )
        print(f"  Agent ready with 3 tools")

        # ── Step 4: Run the agent ──
        print_step(4, "Running agent")
        print(f"  Question: {user_query}\n")

        response = await agent.run(user_query)

        print(f"\n  Answer:\n  {response.text}")

    print(f"\n{'=' * 50}")
    print("  Done!")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    asyncio.run(main())
