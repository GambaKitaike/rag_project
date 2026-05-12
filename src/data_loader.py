import argparse
import json
import pickle
from pathlib import Path

from loaders import build_hyperlink_graph, load_altair_manual_documents


BASE_DIR = Path(__file__).parent.resolve()


def run_crawl(
    start_url: str,
    max_depth: int,
    docs_output: Path,
    graph_output: Path,
) -> None:
    docs = load_altair_manual_documents(start_url, max_depth=max_depth)
    graph = build_hyperlink_graph(docs)

    docs_output.parent.mkdir(parents=True, exist_ok=True)
    graph_output.parent.mkdir(parents=True, exist_ok=True)

    with docs_output.open("wb") as f:
        pickle.dump(docs, f)
    with graph_output.open("w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False, indent=2)

    print(f"Documents saved: {docs_output} ({len(docs)} docs)")
    print(f"Hyperlink graph saved: {graph_output} ({len(graph)} nodes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crawl Altair manual pages and build hyperlink graph.")
    parser.add_argument(
        "--start-url",
        default="https://2021.help.altair.com/2021/hwsolvers/ja_jp/os/index.htm",
        help="Top URL of the manual site.",
    )
    parser.add_argument("--max-depth", type=int, default=8, help="Recursive crawl depth.")
    parser.add_argument(
        "--docs-output",
        default="../data/crawled_documents.pkl",
        help="Output pickle path for crawled langchain Documents.",
    )
    parser.add_argument(
        "--graph-output",
        default="../data/hyperlink_graph.json",
        help="Output JSON path for hyperlink graph.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_crawl(
        start_url=args.start_url,
        max_depth=args.max_depth,
        docs_output=BASE_DIR / args.docs_output,
        graph_output=BASE_DIR / args.graph_output,
    )
