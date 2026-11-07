import os
import sys
import time
import json
from pathlib import Path

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel


def run_docling_test(pdf_path_str: str = None):
    start_time = time.time()
    base_dir = Path(__file__).resolve().parent

    # 1. Resolve PDF Path
    if pdf_path_str:
        pdf_path = Path(pdf_path_str)
    else:
        # Default to the 59-page financial PDF starting with 96051
        candidates = list(base_dir.glob("96051*.pdf")) + list((base_dir / "data").glob("**/96051*.pdf"))
        if candidates:
            pdf_path = candidates[0]
        else:
            print("ERROR: Could not find 96051 PDF file in workspace.")
            sys.exit(1)

    if not pdf_path.exists():
        print(f"ERROR: PDF file not found at: {pdf_path}")
        sys.exit(1)

    print("=" * 70)
    print("DOCLING INDEPENDENT VALIDATION TEST")
    print("=" * 70)
    print(f"Target Document : {pdf_path.name}")
    print(f"File Size       : {pdf_path.stat().st_size / (1024 * 1024):.2f} MB")
    print(f"Absolute Path   : {pdf_path.resolve()}")

    # 2. Setup output directory
    output_dir = base_dir / "docling_test_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_output_path = output_dir / f"{pdf_path.stem}_docling.md"
    json_output_path = output_dir / f"{pdf_path.stem}_docling.json"

    # 3. Process with Docling DocumentConverter
    print("\n[Step 1] Initializing Docling DocumentConverter...")
    init_start = time.time()
    converter = DocumentConverter()
    print(f"  Initialized in {time.time() - init_start:.2f}s")

    print(f"\n[Step 2] Converting '{pdf_path.name}' with DocumentConverter...")
    conv_start = time.time()
    conv_result = converter.convert(str(pdf_path))
    conv_duration = time.time() - conv_start
    print(f"  Conversion completed in {conv_duration:.2f}s ({conv_duration / 60:.2f} minutes)")

    doc = conv_result.document

    # 4. Export Structured Markdown & JSON
    print(f"\n[Step 3] Exporting outputs to '{output_dir.name}'...")
    doc.save_as_markdown(md_output_path)
    print(f"  Saved Structured Markdown : {md_output_path.name} ({md_output_path.stat().st_size / 1024:.1f} KB)")
    
    doc.save_as_json(json_output_path)
    print(f"  Saved Structured JSON     : {json_output_path.name} ({json_output_path.stat().st_size / (1024 * 1024):.2f} MB)")

    # 5. Extract Analytics & Metrics
    print("\n" + "=" * 70)
    print("EXTRACTION & STRUCTURAL AUDIT RESULTS")
    print("=" * 70)

    total_pages = doc.num_pages()
    total_tables = len(doc.tables)
    total_texts = len(doc.texts)

    print(f"Total Pages Processed    : {total_pages}")
    print(f"Total Detected Tables    : {total_tables}")
    print(f"Total Text Elements      : {total_texts}")
    print(f"Total Processing Time    : {conv_duration:.2f}s")

    # Headings extraction
    headings = []
    for item, _ in doc.iterate_items():
        if hasattr(item, "label") and item.label in (DocItemLabel.SECTION_HEADER, DocItemLabel.TITLE):
            text = getattr(item, "text", "").strip()
            page_no = item.prov[0].page_no if hasattr(item, "prov") and item.prov else "N/A"
            if text:
                headings.append((page_no, str(item.label.value), text))

    print(f"\n--- Detected Headings ({len(headings)} total) ---")
    print("Sample Headings (First 15):")
    for page_no, label, h_text in headings[:15]:
        print(f"  [Page {page_no:2}] ({label}): {h_text}")

    # 6. Page 4 Extracted Content
    print("\n" + "=" * 70)
    print("PAGE 4 EXTRACTED CONTENT")
    print("=" * 70)
    page_4_items = []
    page_4_tables = []
    for item, _ in doc.iterate_items():
        if hasattr(item, "prov") and item.prov:
            for p in item.prov:
                if p.page_no == 4:
                    label = getattr(item, "label", "unknown")
                    if hasattr(label, "value"):
                        label = label.value
                    if label == "table":
                        page_4_tables.append(item)
                    else:
                        text = getattr(item, "text", "").strip()
                        if text:
                            page_4_items.append((label, text))
                    break

    print(f"Page 4 Elements: {len(page_4_items)} text blocks, {len(page_4_tables)} tables.\n")
    for label, text in page_4_items:
        print(f"[{label.upper()}]: {text}")

    if page_4_tables:
        print("\n--- Tables Detected on Page 4 ---")
        for idx, tbl in enumerate(page_4_tables, 1):
            print(f"\nTable {idx} on Page 4 (Markdown representation):")
            try:
                tbl_md = tbl.export_to_markdown()
                print(tbl_md)
            except Exception as e:
                print(f"Could not render table as markdown: {e}")

    # 7. Check Consolidated Statement of Financial Position
    print("\n" + "=" * 70)
    print("CONSOLIDATED STATEMENT OF FINANCIAL POSITION AUDIT")
    print("=" * 70)
    target_phrase = "statement of financial position"
    matching_tables = []
    matching_headers = []

    for item, _ in doc.iterate_items():
        text_content = ""
        if hasattr(item, "text"):
            text_content = item.text
        elif hasattr(item, "export_to_markdown"):
            try:
                text_content = item.export_to_markdown()
            except Exception:
                pass

        if target_phrase in text_content.lower():
            label = getattr(item, "label", "unknown")
            if hasattr(label, "value"):
                label = label.value
            page_no = item.prov[0].page_no if hasattr(item, "prov") and item.prov else "N/A"
            if label == "table":
                matching_tables.append((page_no, item))
            else:
                matching_headers.append((page_no, label, text_content.strip()))

    print(f"Mentioned in Headers/Text : {len(matching_headers)} occurrences")
    for p, l, t in matching_headers:
        print(f"  - [Page {p}] ({l}): {t}")

    print(f"\nDetected Financial Position Tables : {len(matching_tables)}")
    table_structure_preserved = False
    if matching_tables:
        for p, tbl in matching_tables:
            print(f"\nTable found on Page {p}:")
            try:
                tbl_md = tbl.export_to_markdown()
                lines = tbl_md.strip().splitlines()
                print(f"  Table Dimensions: {len(lines)} rows in markdown output")
                print("  Sample Table Header & First 8 rows:")
                for l in lines[:10]:
                    print(f"    {l}")
                if len(lines) > 5 and "|" in tbl_md:
                    table_structure_preserved = True
            except Exception as e:
                print(f"  Error exporting table: {e}")
    else:
        # Check if the page containing the statement has tables
        print("Searching for tables on the pages where 'Statement of Financial Position' appears...")
        for p, l, t in matching_headers:
            if isinstance(p, int):
                # check all tables on that page
                for tbl in doc.tables:
                    if tbl.prov and tbl.prov[0].page_no == p:
                        print(f"  Found Table on Page {p}:")
                        tbl_md = tbl.export_to_markdown()
                        for l in tbl_md.strip().splitlines()[:10]:
                            print(f"    {l}")
                        table_structure_preserved = True

    print("\n" + "=" * 70)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 70)
    print(f"1. Docling Installation                   : SUCCESS (docling 2.148.0)")
    print(f"2. Conversion Duration                    : {conv_duration:.2f}s")
    print(f"3. Total Pages Processed                  : {total_pages}")
    print(f"4. Total Tables Detected                  : {total_tables}")
    print(f"5. Headings Extracted                     : {len(headings)}")
    print(f"6. Page 4 Content Extracted               : {'YES' if page_4_items or page_4_tables else 'NO'}")
    print(f"7. Statement of Financial Position Table  : {'PRESERVED' if table_structure_preserved else 'NOT DETECTED AS TABLE'}")
    print(f"8. Markdown Output File                   : {md_output_path.name}")
    print(f"9. JSON Output File                       : {json_output_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_docling_test(arg)

