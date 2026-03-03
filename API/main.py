import argparse
from pathlib import Path

try:
    from .data_quality_checks import run_all_checks
    from .fetch_openagenda import download_csv_export, fetch_records, save_records
    from .preprocess_events import (
        preprocess_raw_csv_to_window_json,
        preprocess_window_json_to_events_jsonl,
    )
    from .rag_pipeline import build_rag_pipeline
    from .ragas_report import run_report, save_report
    from .vectorstore_index import build_vectorstore, load_vectorstore
except ImportError:
    from data_quality_checks import run_all_checks
    from fetch_openagenda import download_csv_export, fetch_records, save_records
    from preprocess_events import (
        preprocess_raw_csv_to_window_json,
        preprocess_window_json_to_events_jsonl,
    )
    from rag_pipeline import build_rag_pipeline
    from ragas_report import run_report, save_report
    from vectorstore_index import build_vectorstore, load_vectorstore


def _cmd_fetch(args):
    if args.mode == "paginated":
        records = fetch_records(region=args.region, limit=args.limit, sleep_s=args.sleep)
        save_records(records)
        print(f"Done. Total records: {len(records)}")
        return
    download_csv_export(region=args.region)


def _cmd_preprocess(args):
    if args.stage in {"window", "all"}:
        preprocess_raw_csv_to_window_json(days_past=args.days_past, days_future=args.days_future)
    if args.stage in {"processed", "all"}:
        preprocess_window_json_to_events_jsonl(days_past=args.days_past, days_future=args.days_future)


def _cmd_index(args):
    if args.mode == "build":
        build_vectorstore(embeddings_model=args.embeddings_model)
        return

    vectorstore, _ = load_vectorstore(embeddings_model=args.embeddings_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})
    results = retriever.invoke(args.query)
    for i, doc in enumerate(results, 1):
        print(f"\n#{i} {doc.metadata.get('title')} | {doc.metadata.get('city')} | {doc.metadata.get('date_start')}")
        print("slug:", doc.metadata.get("slug"))
        print(doc.page_content[:300])


def _cmd_ask(args):
    rag = build_rag_pipeline(search_k=args.search_k, embeddings_model=args.embeddings_model)
    answer, sources = rag.answer(question=args.question, k=args.k, model=args.model, fetch_k=args.fetch_k)
    print("Reponse:\n")
    print(answer)
    print("\nSources:")
    for doc in sources[: args.max_sources]:
        print("-", doc.metadata.get("title"), "|", doc.metadata.get("city"), "|", doc.metadata.get("slug"))


def _cmd_chat(args):
    rag = build_rag_pipeline(search_k=args.search_k, embeddings_model=args.embeddings_model)
    print("Mode chat actif. Tape /quit pour quitter.")
    while True:
        try:
            question = input("\nQuestion > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFin du chat.")
            break

        if not question:
            continue
        if question.lower() in {"/quit", "quit", "exit"}:
            print("Fin du chat.")
            break

        answer, sources = rag.answer(question=question, k=args.k, model=args.model, fetch_k=args.fetch_k)
        print("\nReponse:\n")
        print(answer)
        print("\nSources:")
        for doc in sources[: args.max_sources]:
            print("-", doc.metadata.get("title"), "|", doc.metadata.get("city"), "|", doc.metadata.get("slug"))


def _cmd_report(args):
    questions = None
    if args.questions_file:
        questions = [line.strip() for line in Path(args.questions_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    report_df, sample_df = run_report(questions=questions)
    outputs = save_report(report_df, sample_df)

    print("Ragas report generated:")
    for key, value in outputs.items():
        print(f"- {key}: {value}")
    print("\nSummary:")
    print(report_df[["metric", "description", "category", "status", "mean", "mode"]])


def _cmd_check(args):
    run_all_checks(path=Path(args.path))


def _cmd_pipeline(args):
    # Full happy-path pipeline.
    if args.fetch_mode == "paginated":
        records = fetch_records(region=args.region, limit=args.limit, sleep_s=args.sleep)
        save_records(records)
    else:
        download_csv_export(region=args.region)

    preprocess_raw_csv_to_window_json(days_past=args.days_past, days_future=args.days_future)
    preprocess_window_json_to_events_jsonl(days_past=args.days_past, days_future=args.days_future)
    build_vectorstore(embeddings_model=args.embeddings_model)
    print("Pipeline completed: fetch -> preprocess -> index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PulsEvent command line entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # fetch
    fetch = subparsers.add_parser("fetch", help="Fetch raw OpenAgenda data.")
    fetch.add_argument("--mode", choices=["paginated", "export"], default="paginated")
    fetch.add_argument("--region", default="Normandie")
    fetch.add_argument("--limit", type=int, default=100)
    fetch.add_argument("--sleep", type=float, default=0.2)
    fetch.set_defaults(func=_cmd_fetch)

    # preprocess
    preprocess = subparsers.add_parser("preprocess", help="Preprocess data for RAG.")
    preprocess.add_argument("--stage", choices=["window", "processed", "all"], default="all")
    preprocess.add_argument("--days-past", type=int, default=365)
    preprocess.add_argument("--days-future", type=int, default=365)
    preprocess.set_defaults(func=_cmd_preprocess)

    # index
    index = subparsers.add_parser("index", help="Build or query the FAISS index.")
    index.add_argument("--mode", choices=["build", "search"], default="build")
    index.add_argument("--query", default="concert jazz ce week-end a Rouen")
    index.add_argument("--k", type=int, default=5)
    index.add_argument("--embeddings-model", default="sentence-transformers/all-MiniLM-L6-v2")
    index.set_defaults(func=_cmd_index)

    # ask
    ask = subparsers.add_parser("ask", help="Ask a question to the existing RAG pipeline.")
    ask.add_argument("--question", required=True)
    ask.add_argument("--k", type=int, default=6)
    ask.add_argument("--fetch-k", type=int, default=60)
    ask.add_argument("--search-k", type=int, default=10)
    ask.add_argument("--model", default="mistral-small-latest")
    ask.add_argument("--embeddings-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ask.add_argument("--max-sources", type=int, default=5)
    ask.set_defaults(func=_cmd_ask)

    # chat
    chat = subparsers.add_parser("chat", help="Interactive CLI to ask questions to the RAG pipeline.")
    chat.add_argument("--k", type=int, default=6)
    chat.add_argument("--fetch-k", type=int, default=60)
    chat.add_argument("--search-k", type=int, default=10)
    chat.add_argument("--model", default="mistral-small-latest")
    chat.add_argument("--embeddings-model", default="sentence-transformers/all-MiniLM-L6-v2")
    chat.add_argument("--max-sources", type=int, default=5)
    chat.set_defaults(func=_cmd_chat)

    # report
    report = subparsers.add_parser("report", help="Generate a multi-metric Ragas report.")
    report.add_argument("--questions-file", help="Optional file with one question per line.")
    report.set_defaults(func=_cmd_report)

    # check
    check = subparsers.add_parser("check", help="Run data quality checks.")
    check.add_argument("--path", default=None)
    check.set_defaults(func=_cmd_check)

    # pipeline
    pipeline = subparsers.add_parser("pipeline", help="Run full fetch+preprocess+index pipeline.")
    pipeline.add_argument("--fetch-mode", choices=["paginated", "export"], default="export")
    pipeline.add_argument("--region", default="Normandie")
    pipeline.add_argument("--limit", type=int, default=100)
    pipeline.add_argument("--sleep", type=float, default=0.2)
    pipeline.add_argument("--days-past", type=int, default=365)
    pipeline.add_argument("--days-future", type=int, default=365)
    pipeline.add_argument("--embeddings-model", default="sentence-transformers/all-MiniLM-L6-v2")
    pipeline.set_defaults(func=_cmd_pipeline)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "check" and args.path is None:
        try:
            from .paths import PROCESSED_JSONL_PATH
        except ImportError:
            from paths import PROCESSED_JSONL_PATH
        args.path = str(PROCESSED_JSONL_PATH)
    args.func(args)


if __name__ == "__main__":
    main()
