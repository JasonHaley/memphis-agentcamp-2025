import os
import dotenv

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration,
)
from langchain_text_splitters import MarkdownHeaderTextSplitter

dotenv.load_dotenv(override=True)


def print_step(number, title):
    print(f"\n{'=' * 50}")
    print(f"  STEP {number}: {title}")
    print(f"{'=' * 50}\n")


def chunk_markdown_file(file_path: str) -> list[dict]:
    """Read a markdown file and split it into chunks based on headers."""
    with open(file_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    chunks = splitter.split_text(markdown_content)

    return [
        {
            "index": i,
            "content": chunk.page_content,
            "metadata": chunk.metadata,
            "length": len(chunk.page_content),
        }
        for i, chunk in enumerate(chunks)
    ]


def create_embeddings(text_chunks: list[str], model: str) -> list[list[float]]:
    """Create embeddings for a list of text chunks using Azure OpenAI."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version="2024-02-01",
    )

    response = client.embeddings.create(
        model=model,
        input=text_chunks,
    )

    return [item.embedding for item in response.data]


def ensure_search_index_exists(search_endpoint, search_key, index_name):
    """Create an Azure AI Search index if it doesn't already exist."""
    credential = AzureKeyCredential(search_key)
    index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
    embedding_dimension = 3072

    existing_names = list(index_client.list_index_names())
    if index_name in existing_names:
        print(f"  Index '{index_name}' already exists — skipping creation.")
        return

    print(f"  Creating index '{index_name}'...")

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
            sortable=True,
            facetable=True,
        ),
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
        ),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dimension,
            vector_search_profile_name="chunk-vector-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="chunk-hnsw-config",
                kind="hnsw",
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="chunk-vector-profile",
                algorithm_configuration_name="chunk-hnsw-config",
            )
        ],
    )

    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
    )

    result = index_client.create_index(index)
    print(f"  Index '{result.name}' created.")


def upload_chunks_to_search(text_chunks, embeddings, search_endpoint, search_key, index_name):
    """Upload text chunks and their embeddings to the Azure AI Search index."""
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key),
    )

    docs = [
        {
            "@search.action": "mergeOrUpload",
            "id": str(i),
            "content": text,
            "contentVector": vector,
        }
        for i, (text, vector) in enumerate(zip(text_chunks, embeddings))
    ]

    result = search_client.upload_documents(documents=docs)
    succeeded = sum(1 for r in result if r.succeeded)
    print(f"  Uploaded {succeeded}/{len(docs)} documents.")


def main():
    # ── Config ──
    file_path = "./data/doc2.md"
    randomizer = "jh2"
    index_name = f"{randomizer.lower()}vectorindex"
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_API_KEY")

    # ── Step 1: Chunk the markdown document ──
    print_step(1, "Chunking markdown document")
    chunks = chunk_markdown_file(file_path)
    text_chunks = [item["content"] for item in chunks]
    total_chars = sum(item["length"] for item in chunks)
    avg_size = total_chars / len(chunks) if chunks else 0
    print(f"  Source:  {file_path}")
    print(f"  Chunks:  {len(chunks)}  (avg {avg_size:.0f} chars)")

    # ── Step 2: Generate embeddings ──
    print_step(2, "Generating embeddings")
    embeddings = create_embeddings(text_chunks, model=embedding_model)
    print(f"  Model:      {embedding_model}")
    print(f"  Embeddings: {len(embeddings)}")
    print(f"  Dimensions: {len(embeddings[0])}")

    # ── Step 3: Ensure search index exists ──
    print_step(3, "Ensuring search index exists")
    ensure_search_index_exists(search_endpoint, search_key, index_name)

    # ── Step 4: Upload to search index ──
    print_step(4, "Uploading to search index")
    upload_chunks_to_search(text_chunks, embeddings, search_endpoint, search_key, index_name)

    print(f"\n{'=' * 50}")
    print("  Done! Ingestion pipeline complete.")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
