import asyncio
import os
import dotenv

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from langchain_text_splitters import MarkdownHeaderTextSplitter
from typing import List, Dict

dotenv.load_dotenv(override=True)

def chunk_markdown_file(file_path: str) -> List[Dict]:
    """
    Read a markdown file and split it into chunks based on headers.
    
    Args:
        file_path: Path to the markdown file
    
    Returns:
        List of document chunks with metadata
    """
    
    # Step 1: Read the markdown file
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            markdown_content = file.read()
        print(f"Successfully read file: {file_path}")
        print(f"File size: {len(markdown_content)} characters\n")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    # Step 2: Configure the MarkdownHeaderTextSplitter
    # Define which headers to split on and their metadata keys
    headers_to_split_on = [
        ("#", "Header 1"),      # H1 headers
        ("##", "Header 2"),     # H2 headers
        ("###", "Header 3"),    # H3 headers
    ]
    
    # Create the splitter instance
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False  # Keep headers in the content
    )
    
    # Step 3: Split the document and capture chunks
    chunks = markdown_splitter.split_text(markdown_content)
    
    # Convert to list of dictionaries for easier handling
    chunk_list = []
    for i, chunk in enumerate(chunks):
        chunk_dict = {
            'index': i,
            'content': chunk.page_content,
            'metadata': chunk.metadata,
            'length': len(chunk.page_content)
        }
        chunk_list.append(chunk_dict)
            
    # Return the full list of chunks for use in next lab
    return chunk_list

def output_chunk_stats(chunks):
    if chunks:
        print("\n" + "=" * 60)
        print("📊 SUMMARY STATISTICS:")
        print(f"   Total chunks: {len(chunks)}")
        
        total_chars = sum(chunk['length'] for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        print(f"   Average chunk size: {avg_chunk_size:.1f} characters")
        
        max_chunk = max(chunks, key=lambda x: x['length'])
        min_chunk = min(chunks, key=lambda x: x['length'])
        print(f"   Largest chunk: {max_chunk['length']} characters (chunk #{max_chunk['index']})")
        print(f"   Smallest chunk: {min_chunk['length']} characters (chunk #{min_chunk['index']})")

def create_embeddings(text_chunks: list[str], model: str) -> list[list[float]]:
    # Create a token provider that returns a fresh bearer token on each call
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )

    client = AzureOpenAI(
        azure_ad_token_provider=token_provider,
        api_version="2024-02-01",
    )

    # model is your Azure deployment name, e.g. "embeddings-prod"
    response = client.embeddings.create(
        model=model,          # deployment name, not the base model id[web:42]
        input=text_chunks,    # list of chunk strings
    )

    return [item.embedding for item in response.data]

async def main():
    
    file_path = "./data/doc3.md"

    chunks = chunk_markdown_file(file_path)

    #output_chunk_stats(chunks)

    text_chunks = [item["content"] for item in chunks]

    embeddings = create_embeddings(text_chunks, model=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"))
    print("First embedding:")
    print(embeddings[0])  # a list[float]
    print(len(embeddings[0]), "dimensions")

    print("\nFirst two embeddings:")
    for i, emb in enumerate(embeddings[:2]):  # slice to first 2[web:68][web:72]
        print(f"Embedding {i}:")
        print(emb[:8], "...")  # show just first few dims to keep output short
        print("dim:", len(emb))

if __name__ == "__main__":
    asyncio.run(main())