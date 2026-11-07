from pathlib import Path
from Backend.document_parser import extract_document
from Backend.chunking import create_chunks


def main() -> None:
    print("=" * 60)
    print("Document Chunking Summary (Structure-Aware, 500-char / 100-overlap)")
    print("=" * 60)
    total_chunks = 0
    total_pages = 0

    files = sorted(list(Path("data").glob("**/*.pdf")) + list(Path("data").glob("**/*.docx")))
    for file_path in files[:10]:
        try:
            parsed = extract_document(file_path)
            chunks = create_chunks(parsed, chunk_size=500, chunk_overlap=100)
            print(f"File: {file_path.name}")
            print(f"  Format: {parsed.file_type.upper()} | Pages: {parsed.total_pages} | Chunks: {len(chunks)}")
            print("-" * 40)
            total_chunks += len(chunks)
            total_pages += parsed.total_pages
        except Exception as exc:
            print(f"File {file_path.name}: Error ({exc})")

    print(f"Total Pages Analyzed: {total_pages}")
    print(f"Total Chunks Generated: {total_chunks}")


if __name__ == "__main__":
    main()
