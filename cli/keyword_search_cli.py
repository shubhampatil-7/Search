import argparse
import json
from keyword_search import search_command,build_command, InvertedIndex, tokenize_helper


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a specific document and term")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a specific term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get TF-IDF for a specific document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF for")



    subparsers.add_parser(
        "build",
        help="Build and save the search index",
    )
    args = parser.parse_args()

    match args.command:
        case "search":            
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, res in enumerate(results, 1):
                print(f"{i}. ({res['id']}) {res['title']}")
           
            
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")   
            

        case "tf":
            idx = InvertedIndex()
            idx.load()
            args.term = tokenize_helper(args.term)
            tf = idx.get_tf(args.doc_id, args.term)
            print(f"Term frequency of '{args.term}' in document {args.doc_id}: {tf}")
        
        case "idf":
            idx = InvertedIndex()
            idx.load()
            args.term = tokenize_helper(args.term)
            idf = idx.get_idf(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
            
        case "tfidf":
            idx = InvertedIndex()
            idx.load()
            args.term = tokenize_helper(args.term)
            tf = idx.get_tf(args.doc_id, args.term)
            idf = idx.get_idf(args.term)
            tfidf = tf * idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
