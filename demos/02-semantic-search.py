import os
import dotenv

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

dotenv.load_dotenv(override=True)


def print_step(number, title):
    print(f"\n{'=' * 50}")
    print(f"  STEP {number}: {title}")
    print(f"{'=' * 50}\n")


def get_openai_client() -> AzureOpenAI:
    """Create an Azure OpenAI client using DefaultAzureCredential."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default",
    )
    return AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        azure_ad_token_provider=token_provider,
        api_version="2024-02-01",
    )


def embed_query(client: AzureOpenAI, text: str, model: str) -> list[float]:
    """Embed a single query string."""
    response = client.embeddings.create(model=model, input=[text])
    return response.data[0].embedding


def compare_hybrid_to_vector(
    queries: list[str],
    search_client: SearchClient,
    openai_client: AzureOpenAI,
    embedding_model: str,
    use_hybrid: bool = True,
    top_k: int = 3,
):
    """Run queries against the search index using hybrid or vector-only search."""
    mode = "Hybrid" if use_hybrid else "Vector-only"

    for qi, q in enumerate(queries, 1):
        print()
        print("=" * 80)
        print(f"  Query {qi}/{len(queries)}:  {q}")
        print(f"  Mode: {mode}")
        print("=" * 80)

        q_vector = embed_query(openai_client, q, embedding_model)

        vq = VectorizedQuery(
            vector=q_vector,
            fields="contentVector",
        )

        if use_hybrid:
            results = search_client.search(
                search_text=q,
                vector_queries=[vq],
                top=top_k,
            )
        else:
            results = search_client.search(
                search_text=None,
                vector_queries=[vq],
                top=top_k,
            )

        for rank, doc in enumerate(results, 1):
            score = doc.get("@search.score", None)
            if score is not None:
                print(f"\n  Result {rank}  (id={doc['id']}, score={score:.4f})")
            else:
                print(f"\n  Result {rank}  (id={doc['id']})")
            print(f"  {doc['content']}")
            print(f"  {'─' * 40}")


def main():
    # ── Config ──
    randomizer = "jh"
    index_name = f"{randomizer.lower()}vectorindex"
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_API_KEY")

    queries = [
        #"What are the biggest barriers to AI adoption?",
        #"How does company size affect AI investment?",
        "What percentage of companies use generative AI?",
    ]

    openai_client = get_openai_client()
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key),
    )

    # ── Step 1: Vector-only search ──
    print_step(1, "Vector-only search")
    compare_hybrid_to_vector(
        queries, search_client, openai_client, embedding_model,
        use_hybrid=False, top_k=3,
    )

    # ── Step 2: Hybrid search ──
    print_step(2, "Hybrid search (vector + keyword)")
    compare_hybrid_to_vector(
        queries, search_client, openai_client, embedding_model,
        use_hybrid=True, top_k=3,
    )

    print(f"\n{'=' * 50}")
    print("  Done! Compare the results above.")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
