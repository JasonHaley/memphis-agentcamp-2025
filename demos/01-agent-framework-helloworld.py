import asyncio
import os

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from rich import print

load_dotenv(override=True)

async_credential = None

async_credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(async_credential, "https://cognitiveservices.azure.com/.default")
client = OpenAIChatClient(
    base_url=f"{os.environ['AZURE_OPENAI_ENDPOINT']}/openai/v1/",
    api_key=token_provider,
    model_id=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
)

agent = Agent(client=client, instructions="You're an assistant with worldly knowledge. Answer questions accurately and concisely.")

async def main():
    response = await agent.run("What's the meaning of life?")
    print(response.text)

    if async_credential:
        await async_credential.close()


if __name__ == "__main__":
    asyncio.run(main())
