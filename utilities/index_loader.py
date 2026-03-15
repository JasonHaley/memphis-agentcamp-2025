"""
Azure AI Search Index Loader

This script:
1. Reads a CSV file with support ticket data
2. Creates an Azure AI Search index with the specified schema
3. Generates embeddings for Body and Answer fields using Azure OpenAI
4. Uploads documents to the search index
"""

import csv
import os
import uuid
import time
from typing import List, Dict, Any
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Configuration - Update these values with your Azure resources
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME2 = os.getenv("AZURE_SEARCH_INDEX_NAME2", "support-tickets-index")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME", "text-embedding-3-large")

# Embedding dimension - 1536 for text-embedding-ada-002, 3072 for text-embedding-3-large
EMBEDDING_DIMENSIONS = 3072

# Batch size for uploading documents
BATCH_SIZE = 100


def get_openai_client() -> AzureOpenAI:
    """Create and return an Azure OpenAI client."""

    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION
    )

def get_embeddings(client: AzureOpenAI, text: str) -> List[float]:
    """
    Generate embeddings for the given text using Azure OpenAI.
    
    Args:
        client: Azure OpenAI client
        text: Text to generate embeddings for
        
    Returns:
        List of floats representing the embedding vector
    """
    if not text or text.strip() == "":
        # Return zero vector for empty text
        return [0.0] * EMBEDDING_DIMENSIONS
    
    # Truncate text if too long (max ~8000 tokens for ada-002)
    max_chars = 30000  # Approximate character limit
    if len(text) > max_chars:
        text = text[:max_chars]
    
    try:
        response = client.embeddings.create(
            input=text,
            model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        # Return zero vector on error
        return [0.0] * EMBEDDING_DIMENSIONS


def create_search_index(index_client: SearchIndexClient, index_name: str) -> None:
    """
    Create the Azure AI Search index with the specified schema.
    
    Args:
        index_client: Search index client
        index_name: Name of the index to create
    """
    # Define vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw-config",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine"
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile",
                algorithm_configuration_name="hnsw-config"
            )
        ]
    )
    
    # Define the index fields
    fields = [
        SimpleField(
            name="Id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        SimpleField(
            name="Create_Date", 
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True, 
            sortable=True, 
            facetable=False
        ),
        SearchableField(
            name="Subject",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            sortable=True
        ),
        SearchableField(
            name="Body",
            type=SearchFieldDataType.String,
            searchable=True
        ),
        SearchableField(
            name="Answer",
            type=SearchFieldDataType.String,
            searchable=True
        ),
        SimpleField(
            name="Type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SimpleField(
            name="Queue",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SimpleField(
            name="Priority",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
            sortable=True
        ),
        SimpleField(
            name="Language",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SimpleField(
            name="Business_Type",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        SearchField(
            name="Tags",
            type=SearchFieldDataType.Collection(SearchFieldDataType.String),
            filterable=True,
            facetable=True
        ),
        SearchField(
            name="BodyEmbeddings",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="vector-profile"
        ),
        SearchField(
            name="AnswerEmbeddings",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=EMBEDDING_DIMENSIONS,
            vector_search_profile_name="vector-profile"
        ),    
    ]
    
    # Create the index
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    
    try:
        # Delete index if it already exists
        index_client.delete_index(index_name)
        print(f"Deleted existing index: {index_name}")
    except Exception:
        pass  # Index doesn't exist, which is fine
    
    index_client.create_index(index)
    print(f"Created index: {index_name}")


def collect_tags(row: Dict[str, str]) -> List[str]:
    """
    Collect all non-empty tags from the CSV row into a list.
    
    Args:
        row: CSV row dictionary
        
    Returns:
        List of non-empty tag values
    """
    tags = []
    for i in range(1, 10):  # tag_1 through tag_9
        tag_key = f"tag_{i}"
        if tag_key in row and row[tag_key] and row[tag_key].strip():
            tags.append(row[tag_key].strip())
    return tags


def load_csv_file(file_path: str) -> List[Dict[str, str]]:
    """
    Load and parse the CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        List of dictionaries representing each row
    """
    rows = []
    with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(row)
    print(f"Loaded {len(rows)} rows from CSV file")
    return rows


def transform_row_to_document(
    row: Dict[str, str],
    openai_client: AzureOpenAI,
    row_index: int
) -> Dict[str, Any]:
    """
    Transform a CSV row into a search document with embeddings.
    
    Args:
        row: CSV row dictionary
        openai_client: Azure OpenAI client for generating embeddings
        row_index: Index of the row for generating unique ID
        
    Returns:
        Dictionary representing the search document
    """
    # Generate unique ID using UUID
    doc_id = str(uuid.uuid4())
    
    # Get text fields
    body = row.get('body', '') or ''
    answer = row.get('answer', '') or ''
    
    # Generate embeddings
    body_embeddings = get_embeddings(openai_client, body)
    answer_embeddings = get_embeddings(openai_client, answer)
    
    # Collect tags
    tags = collect_tags(row)
    
    document = {
        "Id": doc_id,
        "Create_Date": row.get('create_date', '') or '',
        "Subject": row.get('subject', '') or '',
        "Body": body,
        "Answer": answer,
        "Type": row.get('type', '') or '',
        "Queue": row.get('queue', '') or '',
        "Priority": row.get('priority', '') or '',
        "Language": row.get('language', '') or '',
        "Business_Type": row.get('business_type', '') or '',
        "Tags": tags,
        "BodyEmbeddings": body_embeddings,
        "AnswerEmbeddings": answer_embeddings
    }
    
    return document


def upload_documents(
    search_client: SearchClient,
    documents: List[Dict[str, Any]]
) -> None:
    """
    Upload documents to Azure AI Search in batches.
    
    Args:
        search_client: Search client for uploading documents
        documents: List of documents to upload
    """
    total = len(documents)
    uploaded = 0
    
    for i in range(0, total, BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        try:
            result = search_client.upload_documents(documents=batch)
            succeeded = sum(1 for r in result if r.succeeded)
            uploaded += succeeded
            print(f"Uploaded batch {i // BATCH_SIZE + 1}: {succeeded}/{len(batch)} documents succeeded")
        except Exception as e:
            print(f"Error uploading batch: {e}")
    
    print(f"Total uploaded: {uploaded}/{total} documents")


def main(csv_file_path: str):
    """
    Main function to orchestrate the index creation and data loading.
    
    Args:
        csv_file_path: Path to the CSV file to load
    """
    print("=" * 60)
    print("Azure AI Search Index Loader")
    print("=" * 60)
        
    # Initialize clients
    print("\nInitializing clients...")
    
    search_credential = AzureKeyCredential(AZURE_SEARCH_API_KEY)
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=search_credential
    )
    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX_NAME2,
        credential=search_credential
    )
    openai_client = get_openai_client()
    
    # Create the search index
    print("\nCreating search index...")
    create_search_index(index_client, AZURE_SEARCH_INDEX_NAME2)
    
    # Wait a moment for index to be ready
    time.sleep(2)
    
    # Load CSV file
    print(f"\nLoading CSV file: {csv_file_path}")
    rows = load_csv_file(csv_file_path)
    
    # Transform rows to documents with embeddings
    print("\nTransforming rows and generating embeddings...")
    documents = []
    for i, row in enumerate(rows):
        if (i + 1) % 10 == 0:
            print(f"  Processing row {i + 1}/{len(rows)}...")
        doc = transform_row_to_document(row, openai_client, i)
        documents.append(doc)
    
    # Upload documents
    print("\nUploading documents to search index...")
    upload_documents(search_client, documents)
    
    print("\n" + "=" * 60)
    print("Data loading complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python index_loader.py <path_to_csv_file>")
        print("\nExample:")
        print("  python index_loader.py data/support_tickets.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    main(csv_path)
