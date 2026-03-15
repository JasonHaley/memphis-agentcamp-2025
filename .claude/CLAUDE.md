# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Demo scripts for the Memphis AgentCamp 2025 presentation, showcasing a progression from basic AI agents to full agentic RAG pipelines using Azure OpenAI and the `agent-framework` SDK.

## Setup & Commands

- **Python**: 3.13 (managed via `uv`, see `.python-version`)
- **Install dependencies**: `uv sync`
- **Run a demo**: `uv run python demos/<script>.py` (each demo is a standalone script)
- **Convert PDF to markdown**: `uv run python utilities/pdf_to_markdown.py <input.pdf> [output.md]`

## Environment

Requires a `.env` file with Azure OpenAI and Azure AI Search credentials. Uses `DefaultAzureCredential` for auth (Azure CLI login or managed identity). Key env vars:
- `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_CHAT_DEPLOYMENT` — chat model
- `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` — embedding model (used in ingestion)
- `AZURE_SEARCH_ENDPOINT`, `AZURE_SEARCH_KNOWLEDGE_BASE_NAME` — Azure AI Search (demos 04+)

## Architecture

The demos follow a numbered progression building toward agentic RAG:

1. **01-agent-framework-helloworld** — Minimal `agent_framework.Agent` + `OpenAIChatClient` with Azure AD auth
2. **02-ingestion-phase** — PDF-to-markdown chunking (via `langchain_text_splitters.MarkdownHeaderTextSplitter`) and embedding generation with Azure OpenAI
3. **03-simple-rag** — Basic retrieval-augmented generation (placeholder)
4. **04-advanced-rag** — Advanced RAG patterns (placeholder)
5. **05-agentic-rag** — Agentic RAG (placeholder)
6. **06-agentic-rag-with-kb** — Agentic RAG with Azure AI Search knowledge base (placeholder)

Key libraries: `agent-framework-core`, `agent-framework-devui`, `agent-framework-orchestrations`, `agent-framework-azure-ai-search`, `openai`, `langchain-text-splitters`, `markitdown`, `pypdf`.

## Utilities

- `utilities/pdf_to_markdown.py` — Uses `markitdown` library for quick PDF conversion
- `utilities/pdf_to_markdown-manual.py` — Uses `pdfplumber`/`pypdf` with heuristic heading detection, table extraction, and metadata handling

## Data

`data/` contains source PDFs/docx and pre-converted markdown files (`doc1.md`, `doc2.md`, `doc3.md`) used as input for the ingestion and RAG demos.
