import os
from time import sleep
from typing import Literal, NotRequired

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from search_utils import SearchResult
import re


class RerankedSearchResult(SearchResult, total=False):
    individual_score: NotRequired[int]


load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    max_retries=0,
)
model = "openrouter/free"


def llm_rerank_individual(
    query: str, documents: list[SearchResult], limit: int = 5
) -> list[RerankedSearchResult]:
    scored_docs: list[RerankedSearchResult] = []

    for doc in documents:
        prompt = f"""Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Output ONLY the number in your response, no other text or explanation.

        Score:"""

        max_attempts = 3
        score_text = ""

        for attempt in range(1, max_attempts + 1):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                score_text = (response.choices[0].message.content or "").strip()

                if re.fullmatch(r"(?:10|[0-9])", score_text):
                    score = int(score_text)
                    break
            except (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            ) as exc:
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Reranking request failed after {max_attempts} attempts "
                        f"for {doc.get('title', '<untitled>')!r}"
                    ) from exc

            if attempt < max_attempts:
                sleep(attempt)  # waits 1 second, then 2 seconds
        else:
            raise ValueError(
                f"Expected a score from 0 to 10 for {doc.get('title', '<untitled>')!r} "
                f"after {max_attempts} attempts; last response was {score_text!r}"
            )

        scored_docs.append({**doc, "individual_score": score})
        sleep(3)

    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True) # type: ignore
    return scored_docs[:limit]


def rerank(
    query: str,
    documents: list[SearchResult],
    method: Literal["individual", "batch", "cross_encoder"] = "batch",
    limit: int = 5,
) -> list[SearchResult] | list[RerankedSearchResult]:
    if method == "individual":
        return llm_rerank_individual(query, documents, limit)
    else:
        return documents[:limit]
