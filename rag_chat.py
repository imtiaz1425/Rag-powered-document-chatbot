import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

from Backend.rag_service import DocumentChatService


def main():
    print("=" * 60)
    print("AI Document Chatbot — CLI Interactive Interface")
    print("=" * 60)

    service = DocumentChatService()

    # Find available documents
    upload_dir = BASE_DIR / "data" / "uploads"
    reports_dir = BASE_DIR / "data" / "reports"
    test_dir = BASE_DIR / "data" / "test_docs"

    all_docs = []
    for d in [upload_dir, test_dir, reports_dir]:
        if d.exists():
            all_docs.extend(list(d.glob("*.pdf")) + list(d.glob("*.docx")))

    if not all_docs:
        print("No documents found in data/ directories.")
        doc_path_str = input("Enter path to a PDF or Word document: ").strip()
        doc_path = Path(doc_path_str)
    else:
        print("\nAvailable documents:")
        for idx, p in enumerate(all_docs[:8], 1):
            print(f"  [{idx}] {p.name} ({p.parent.name})")

        choice = input(f"\nSelect a document [1-{min(len(all_docs), 8)}] or enter a file path: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= min(len(all_docs), 8):
            doc_path = all_docs[int(choice) - 1]
        else:
            doc_path = Path(choice)

    if not doc_path.exists():
        print(f"Error: File '{doc_path}' does not exist.")
        sys.exit(1)

    print(f"\nProcessing and indexing '{doc_path.name}'...")
    try:
        res = service.process_and_index_document(doc_path)
        print(f"Ready: {res['pages']} pages, {res['chunks']} chunks indexed with FAISS + BM25.\n")
        print("You can now ask questions about this document. Type 'exit' to quit.\n")
    except Exception as e:
        print(f"Error indexing document: {e}")
        sys.exit(1)

    while True:
        try:
            query = input("Ask a question: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("Exiting CLI Chatbot. Goodbye!")
                break

            response = service.answer_question(query)
            print("\n" + "-" * 50)
            print("ANSWER:")
            print(response.get("answer", "No answer."))
            pages = response.get("pages", [])
            if pages:
                print("\nSOURCES:")
                print("Cited Pages:", ", ".join(f"Page {p}" for p in pages))
            print("-" * 50 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break


if __name__ == "__main__":
    main()
