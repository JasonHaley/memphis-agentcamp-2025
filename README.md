# Memphis AgentCamp 2025 - Agentic RAG Demos

Demo scripts for the [Memphis AgentCamp](https://www.meetup.com/memphis-technology-user-groups/events/312731040/) Agentic RAG presentation, walking through a hands-on progression from document ingestion to fully agentic RAG pipelines using Azure OpenAI and the `agent-framework` SDK.

## Demos

The demos are numbered to follow the presentation flow, each building on concepts from the previous step.

| # | Demo | Description |
|---|------|-------------|
| 1 | [Ingestion Phase](demos/01-ingestion-phase.py) | Chunks markdown documents, generates embeddings via Azure OpenAI, and uploads them to an Azure AI Search index. |
| 2 | [Semantic Search](demos/02-semantic-search.py) | Runs vector-only and hybrid search queries against the index, comparing how different search modes rank results. |
| 3 | [Simple RAG](demos/03-simple-rag.py) | A basic retrieval-augmented generation pipeline that retrieves context chunks and generates answers with source citations. |
| 4 | [Agent Framework Hello World](demos/04-agent-framework-helloworld.py) | Minimal example using the `agent-framework` library with Azure OpenAI and Azure AD authentication. |
| 5 | [Agentic RAG](demos/05-agentic-rag.py) | An agent that autonomously calls multiple tools (knowledge base search, support tickets, Bing grounding) to answer questions via multi-hop reasoning. |
| 6 | [Agentic RAG with Knowledge Base](demos/06-agentic-rag-with-kb.py) | Uses `AzureAISearchContextProvider` in agentic mode, letting the agent plan and execute multi-hop queries across a knowledge base with multi-turn conversation context. |

## Utilities

| Utility | Description |
|---------|-------------|
| [pdf_to_markdown.py](utilities/pdf_to_markdown.py) | Lightweight PDF-to-Markdown converter using the `markitdown` library. |
| [pdf_to_markdown-manual.py](utilities/pdf_to_markdown-manual.py) | More sophisticated converter using `pdfplumber`/`pypdf` with heuristic heading detection and table extraction. |
| [index_loader.py](utilities/index_loader.py) | Reads support ticket data from CSV, creates an Azure AI Search index with dual embeddings, and uploads documents for the agentic RAG demo. |

## Getting Started

```bash
# Install dependencies (requires uv and Python 3.13)
uv sync

# Run any demo
uv run python demos/01-ingestion-phase.py
```

Requires a `.env` file with Azure OpenAI and Azure AI Search credentials. See [CLAUDE.md](.claude/CLAUDE.md) for the full list of required environment variables.
