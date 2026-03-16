import os
import dotenv

from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery

dotenv.load_dotenv(override=True)

SYSTEM_PROMPT = """
You are an AI assistant specializing in generative AI topics such as large language models, prompt engineering, fine-tuning, RAG, AI agents, and related techniques.
Answer the question using only the provided context.
Use bullets if the answer has multiple points.
If the answer is longer than 3 sentences, provide a summary.
Answer ONLY with the facts listed in the list of sources provided in the context with the user query.
Cite your source when you answer the question with the format [source-id].
If the answer is not contained within the context, respond with "I don't know."
"""

RAG_PROMPT = """
User Question: {user_query}
Context:
{context}
"""


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


def retrieve_context(
    query: str,
    search_client: SearchClient,
    openai_client: AzureOpenAI,
    embedding_model: str,
    use_hybrid: bool = True,
    top_k: int = 3,
) -> list[dict]:
    """Search the index and return the top-k results as context chunks."""
    q_vector = embed_query(openai_client, query, embedding_model)

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

    return [
        {
            "id": doc["id"],
            "content": doc["content"],
            "score": doc.get("@search.score"),
        }
        for doc in results
    ]


def build_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string with source IDs."""
    parts = []
    for chunk in chunks:
        parts.append(f"[source-{chunk['id']}]\n{chunk['content']}")
    return "\n\n".join(parts)


def generate_answer(
    openai_client: AzureOpenAI,
    chat_model: str,
    user_query: str,
    context: str,
) -> str:
    """Send the RAG prompt to the chat model and return the response."""
    user_message = RAG_PROMPT.format(user_query=user_query, context=context)

    response = openai_client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.choices[0].message.content


def retrieve_context_text_only(
    query: str,
    search_client: SearchClient,
    top_k: int = 3,
) -> list[dict]:
    """Search the index using text-only (keyword) search — no vectors."""
    results = search_client.search(
        search_text=query,
        top=top_k,
    )

    return [
        {
            "id": doc["id"],
            "content": doc["content"],
            "score": doc.get("@search.score"),
        }
        for doc in results
    ]


def run_rag(
    user_query: str,
    search_client: SearchClient,
    openai_client: AzureOpenAI,
    embedding_model: str,
    chat_model: str,
    use_hybrid: bool,
    use_text_only: bool = False,
    top_k: int = 3,
    show_search_results: bool = True,
):
    """Run the full RAG pipeline: retrieve, build context, generate answer."""
    if use_text_only:
        mode = "Text-only"
    elif use_hybrid:
        mode = "Hybrid"
    else:
        mode = "Vector-only"

    # ── Retrieve ──
    print_step(1, f"Retrieving context ({mode} search)")
    if use_text_only:
        chunks = retrieve_context_text_only(
            user_query, search_client, top_k=top_k,
        )
    else:
        chunks = retrieve_context(
            user_query, search_client, openai_client, embedding_model,
            use_hybrid=use_hybrid, top_k=top_k,
        )
    print(f"  Retrieved {len(chunks)} chunks.")
    if show_search_results:
        for rank, chunk in enumerate(chunks, 1):
            score = chunk["score"]
            if score is not None:
                print(f"\n  Result {rank}  (id={chunk['id']}, score={score:.4f})")
            else:
                print(f"\n  Result {rank}  (id={chunk['id']})")
            print(f"  {chunk['content']}")
            print(f"  {'─' * 40}")

    # ── Build context ──
    context = build_context(chunks)

    # ── Generate ──
    print_step(2, "Generating answer")
    print(f"  Model: {chat_model}")
    print(f"  Mode:  {mode}\n")

    answer = generate_answer(openai_client, chat_model, user_query, context)

    print(f"  Question: {user_query}\n")
    print(f"  Answer:\n  {answer}")


def main():
    # ── Config ──
    randomizer = "jh"
    index_name = f"{randomizer.lower()}vectorindex"
    embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    chat_model = os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT2"]
    search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
    search_key = os.getenv("AZURE_SEARCH_API_KEY")

    #"What percentage of companies use generative AI?"
    user_query = "Can you explain why only 5% of custom enterprise AI projects succeed?"
    show_search_results = False  # Set to False to hide retrieved chunks

    openai_client = get_openai_client()
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(search_key),
    )

    # ── Option A: Vector-only RAG ──
    print("\n" + "█" * 50)
    print("  OPTION A: RAG with Vector-only search")
    print("█" * 50)
    run_rag(
        user_query, search_client, openai_client,
        embedding_model, chat_model, use_hybrid=False,
        show_search_results=show_search_results,
    )

    # ── Option B: Hybrid RAG ──
    print("\n\n" + "█" * 50)
    print("  OPTION B: RAG with Hybrid search")
    print("█" * 50)
    run_rag(
        user_query, search_client, openai_client,
        embedding_model, chat_model, use_hybrid=True,
        show_search_results=show_search_results,
    )

    # ── Option C: Text-only RAG ──
    print("\n\n" + "█" * 50)
    print("  OPTION C: RAG with Text-only (keyword) search")
    print("█" * 50)
    run_rag(
        user_query, search_client, openai_client,
        embedding_model, chat_model, use_hybrid=False,
        use_text_only=True,
        show_search_results=show_search_results,
    )

    print(f"\n{'=' * 50}")
    print("  Done! Compare the three answers above.")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
