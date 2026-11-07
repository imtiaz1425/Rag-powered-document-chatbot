from pathlib import Path
from Backend.document_parser import extract_document


def main() -> None:
    print("=" * 60)
    print("Document Text & Table Extraction Inspector")
    print("=" * 60)
    for file_path in sorted(list(Path("data").glob("**/*.pdf")) + list(Path("data").glob("**/*.docx")))[:6]:
        try:
            doc = extract_document(file_path)
            total_chars = sum(len(p.text) for p in doc.pages)
            scanned_pages = sum(1 for p in doc.pages if p.is_scanned)
            print(f"File: {file_path.name}")
            print(f"  Type: {doc.file_type.upper()} | Pages: {doc.total_pages} | Total Characters: {total_chars}")
            print(f"  Scanned / Graphic Pages: {scanned_pages}")
            p1 = doc.pages[0]
            print(f"  Page 1 Preview: {repr(p1.text[:120])}...")
            print("-" * 50)
        except Exception as exc:
            print(f"Error reading {file_path.name}: {exc}")


if __name__ == "__main__":
    main()
