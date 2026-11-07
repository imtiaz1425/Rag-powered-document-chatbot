import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from Backend.document_parser import extract_document
from Backend.chunking import create_chunks
from Backend.retriever import HybridRetriever


def main():
    print("=" * 60)
    print("Hybrid Retrieval Tester (FAISS + BM25 + RRF)")
    print("=" * 60)

    # Pick a document
    doc_path = BASE_DIR / "data" / "uploads" / "bank of america 10kreports.pdf"
    if not doc_path.exists():
        candidates = list((BASE_DIR / "data").glob("**/*.pdf")) + list((BASE_DIR / "data").glob("**/*.docx"))
        if candidates:
            doc_path = candidates[0]
        else:
            print("No documents found to test.")
            return

    print(f"Loading document: {doc_path.name}")
    parsed = extract_document(doc_path)
    chunks = create_chunks(parsed, chunk_size=500, chunk_overlap=100)
    print(f"Indexed {len(chunks)} chunks. Initializing HybridRetriever...")
    retriever = HybridRetriever(chunks)

    while True:
        query = input("\nEnter query (or 'exit' to quit): ").strip()
        if not query or query.lower() in ("exit", "quit"):
            break

        results = retriever.retrieve(query, top_k=5)
        print(f"\nTop {len(results)} Hybrid Retrieved Chunks for: '{query}'")
        print("=" * 60)
        for i, chunk in enumerate(results, 1):
            print(f"[Result {i}] Page {chunk.page_number} | Section: {chunk.section_heading} | ID: {chunk.chunk_id}")
            print(f"Content:\n{chunk.content}")
            print("-" * 60)


if __name__ == "__main__":
    main()
