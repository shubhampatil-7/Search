import argparse
from lib.semantic_search import SemanticSearch, embed_text, verify_embeddings, embed_query_text

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

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()