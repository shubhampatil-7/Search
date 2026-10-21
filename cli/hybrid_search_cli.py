import argparse
from hybrid_search import normalize


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help = "normalize a list of score to 0-1")
    normalize_parser.add_argument("list", type=float, nargs="+", help = "provide a list of scores to normalize")

    args = parser.parse_args()

    

    match args.command:
        case "normalize":
            scores : list[int]= args.list
            res = normalize(scores)
            if not res or len(res) == 0:
                print("")
            else:    
                for score in res:
                    print(f"* {score:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()