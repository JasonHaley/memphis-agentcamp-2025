import asyncio
import os

from agent_framework import Agent
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
    question = "What's the meaning of life?"

    # ── Step 1: Set up the AI client ──
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

    # ── Step 2: Create an agent and ask a question ──
    print_step(2, "Running agent")
    agent = Agent(
        client=client,
        instructions="You're an assistant with worldly knowledge. Answer questions accurately and concisely.",
    )
    print(f"  Question: {question}\n")

    response = await agent.run(question)
    print(f"  Answer: {response.text}")

    await credential.close()

    print(f"\n{'=' * 50}")
    print("  Done!")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    asyncio.run(main())
