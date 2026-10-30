from search_utils import RRF_K, load_movies
from hybrid_search import rrf_search_command

import os
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
model = "openrouter/free"

def augment(query, limit):
    movies = load_movies()
    res = rrf_search_command(query, RRF_K, None, limit)
    docs = res["results"]

    prompt = f"""You are a RAG agent for Webflyx, a movie streaming service.
        Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
        Provide a comprehensive answer that addresses the user's query.

        Query: {query}

        Documents:
        {docs}

        Answer:"""

    response = client.chat.completions.create( model=model, messages=[{"role": "user", "content": prompt}]
    )

    llm_res = (response.choices[0].message.content or "").strip().strip('"')

    print("Search Results:\n")
    for i in res["results"]:
        print(f"- {i['title']}")
    print("\n")
    print("RAG Response:\n")
    print(llm_res)