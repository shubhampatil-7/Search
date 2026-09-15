import argparse
from lib.semantic_search import SemanticSearch, embed_text, verify_embeddings, embed_query_text, semantic_search, chunk_text
from search_utils import CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command")

    verify_parser = subparsers.add_parser(
        "verify",
        help="Verify semantic search functionality"
    )

    embed_text_parser = subparsers.add_parser("embed_text", help="Generate embedding for a given text")
    embed_text_parser.add_argument("text", type=str, help="Text to generate embedding for")
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Verify embeddings functionality")

    embed_query_parser = subparsers.add_parser("embed_query", help="Generate embedding for a given query")
    embed_query_parser.add_argument("query", type=str, help="Query to generate embedding for")
    
    search_parser = subparsers.add_parser("search", help="Search for relevant documents based on a query")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    chunk_parser = subparsers.add_parser("chunk", help="Chunk a doc into smaller chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="Size of each chunk")
    chunk_parser.add_argument("--overlap", type=int, default = DEFAULT_CHUNK_OVERLAP, help="Size of each chunk")

    args = parser.parse_args()
    
    match args.command:
        case "verify":
            ss = SemanticSearch()
            ss.verify()
        
        case "embed_text":
            text = args.text
            embed_text(text)
        
        case "verify_embeddings":
            verify_embeddings()
        case "embed_query":
            query = args.query
            embed_query_text(query)

        case "search":
            semantic_search(args.query, args.limit)

        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()