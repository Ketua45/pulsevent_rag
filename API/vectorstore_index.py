import argparse
import json

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    from .paths import PROCESSED_JSONL_PATH, VECTORSTORE_DIR
except ImportError:
    from paths import PROCESSED_JSONL_PATH, VECTORSTORE_DIR


def load_documents_from_jsonl(jsonl_path=PROCESSED_JSONL_PATH):
    docs = []
    event_catalog = {}
    event_catalog_by_slug = {}

    with open(jsonl_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            event_id = str(record.get("event_id", "")).strip()
            slug = str(record.get("slug", "")).strip()
            if event_id:
                event_catalog[event_id] = record
            if slug:
                event_catalog_by_slug[slug] = record

            docs.append(
                Document(
                    page_content=record["document_text"],
                    metadata={
                        "event_id": event_id,
                        "title": record.get("title", ""),
                        "location_name": record.get("location_name", ""),
                        "address": record.get("address", ""),
                        "postal_code": record.get("postal_code", ""),
                        "city": record.get("city", ""),
                        "department": record.get("department", ""),
                        "date_start": record.get("date_start", ""),
                        "date_end": record.get("date_end", ""),
                        "lat": record.get("lat", None),
                        "lon": record.get("lon", None),
                        "slug": slug,
                        "source": record.get("source", "openagenda"),
                    },
                )
            )

    return docs, event_catalog, event_catalog_by_slug


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(
    jsonl_path=PROCESSED_JSONL_PATH,
    vectorstore_dir=VECTORSTORE_DIR,
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    docs, event_catalog, event_catalog_by_slug = load_documents_from_jsonl(jsonl_path)
    chunks = split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(vectorstore_dir))
    print(f"FAISS index saved to: {vectorstore_dir}")
    return vectorstore, embeddings, event_catalog, event_catalog_by_slug


def load_vectorstore(
    vectorstore_dir=VECTORSTORE_DIR,
    embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    vectorstore = FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or search the FAISS vectorstore.")
    parser.add_argument("--mode", choices=["build", "search"], default="build")
    parser.add_argument("--query", default="concert jazz ce week-end a Rouen")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.mode == "build":
        build_vectorstore()
        return

    vectorstore, _ = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": args.k})
    results = retriever.invoke(args.query)
    for i, doc in enumerate(results, 1):
        print(f"\n#{i} {doc.metadata.get('title')} | {doc.metadata.get('city')} | {doc.metadata.get('date_start')}")
        print("slug:", doc.metadata.get("slug"))
        print(doc.page_content[:300])


if __name__ == "__main__":
    main()
