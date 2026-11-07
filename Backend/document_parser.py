import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pymupdf  # PyMuPDF v2.1+
import docx  # python-docx


class ExtractedPage:
    def __init__(self, page_number: int, text: str, tables: Optional[List[str]] = None, headings: Optional[List[str]] = None, is_scanned: bool = False, elements: Optional[List[Dict[str, Any]]] = None):
        self.page_number = page_number
        self.text = text.strip()
        self.tables = tables or []
        self.headings = headings or []
        self.is_scanned = is_scanned
        self.elements = elements or []

    def get_full_content(self) -> str:
        """Combine headings, tables, and page text into structured content."""
        return self.text


class ParsedDocument:
    def __init__(self, filename: str, file_type: str, total_pages: int, pages: List[ExtractedPage], metadata: Optional[Dict[str, Any]] = None):
        self.filename = filename
        self.file_type = file_type
        self.total_pages = total_pages
        self.pages = pages
        self.metadata = metadata or {}


def table_to_markdown(table_data: List[List[Any]]) -> str:
    """Convert a 2D list of cells into a clean Markdown table."""
    if not table_data or len(table_data) == 0:
        return ""
    
    cleaned = []
    for row in table_data:
        cleaned_row = [re.sub(r"\s+", " ", str(cell or "").strip()) for cell in row]
        if any(cleaned_row):
            cleaned.append(cleaned_row)
            
    if not cleaned:
        return ""

    num_cols = max(len(row) for row in cleaned)
    for row in cleaned:
        while len(row) < num_cols:
            row.append("")

    header = cleaned[0]
    md_lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * num_cols) + " |"
    ]
    for row in cleaned[1:]:
        md_lines.append("| " + " | ".join(row) + " |")

    return "\n".join(md_lines)


def format_tabular_block(text: str) -> str:
    """
    Detect if a block has tabular characteristics (multiple columns separated by 2+ spaces),
    and if so, normalize it cleanly so LLM and embeddings recognize the tabular relationships.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return text

    table_rows = []
    is_table = True
    for line in lines:
        parts = [p.strip() for p in re.split(r"\s{2,}|\t+", line) if p.strip()]
        if len(parts) >= 2:
            table_rows.append(parts)
        else:
            is_table = False
            break

    if is_table and len(table_rows) >= 2:
        max_cols = max(len(r) for r in table_rows)
        if max_cols >= 2:
            return table_to_markdown(table_rows)

    return text


def parse_pdf(file_path: Path) -> ParsedDocument:
    """
    Extract structured text, headings, and tables from a PDF using PyMuPDF blocks.
    Preserves reading order, identifies headings, handles columns, and formats tables.
    """
    doc = pymupdf.open(str(file_path))
    total_pages = len(doc)
    extracted_pages: List[ExtractedPage] = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]
        headings = []
        is_scanned = False
        text_blocks = []

        # Extract blocks: (x0, y0, x1, y1, text, block_no, block_type)
        blocks = page.get_text("blocks")
        # Sort blocks in visual reading order: top-to-bottom, left-to-right
        blocks = sorted(blocks, key=lambda b: (round(b[1] / 15) * 15, b[0]))

        for b in blocks:
            if b[6] == 0:  # text block
                b_text = b[4].strip()
                if not b_text:
                    continue
                # Heading detection: single line, no ending period, relatively short
                lines = b_text.splitlines()
                if len(lines) == 1 and len(b_text) < 90 and not b_text.endswith("."):
                    headings.append(b_text)
                    text_blocks.append(f"### {b_text}")
                else:
                    formatted_block = format_tabular_block(b_text)
                    text_blocks.append(formatted_block)

        raw_page_text = "\n\n".join(text_blocks).strip()

        # Scanned page detection
        if len(raw_page_text) < 40:
            images = page.get_images()
            if images:
                is_scanned = True
                raw_page_text = f"[Page {page_num}: Scanned image-based content without embedded text layer.]"

        extracted_pages.append(ExtractedPage(
            page_number=page_num,
            text=raw_page_text,
            headings=headings,
            is_scanned=is_scanned
        ))

    doc.close()

    return ParsedDocument(
        filename=file_path.name,
        file_type="pdf",
        total_pages=total_pages,
        pages=extracted_pages,
        metadata={"title": file_path.stem, "parser_used": "pymupdf"}
    )


def parse_docx(file_path: Path) -> ParsedDocument:
    """
    Extract structured text, headings, lists, and tables from a Word (.docx) document.
    """
    doc = docx.Document(str(file_path))
    current_page_blocks = []
    current_headings = []
    pages: List[ExtractedPage] = []
    page_counter = 1
    char_count = 0
    CHARS_PER_PAGE = 2500

    def flush_page():
        nonlocal current_page_blocks, current_headings, page_counter, char_count
        if current_page_blocks:
            text = "\n\n".join(current_page_blocks).strip()
            pages.append(ExtractedPage(
                page_number=page_counter,
                text=text,
                headings=list(current_headings)
            ))
            page_counter += 1
            current_page_blocks = []
            current_headings = []
            char_count = 0

    for elem in doc.element.body:
        if elem.tag.endswith('p'):
            for p in doc.paragraphs:
                if p._element == elem:
                    text = p.text.strip()
                    if not text:
                        continue
                    style_name = p.style.name.lower() if p.style else ""
                    if "heading" in style_name:
                        level = 1
                        match = re.search(r"\d+", style_name)
                        if match:
                            level = min(int(match.group()), 6)
                        md_heading = f"{'#' * level} {text}"
                        current_page_blocks.append(md_heading)
                        current_headings.append(text)
                    elif "list" in style_name or "bullet" in style_name:
                        current_page_blocks.append(f"- {text}")
                    else:
                        current_page_blocks.append(text)

                    char_count += len(text)
                    if char_count >= CHARS_PER_PAGE:
                        flush_page()
                    break

        elif elem.tag.endswith('tbl'):
            for t in doc.tables:
                if t._element == elem:
                    table_matrix = []
                    for row in t.rows:
                        row_data = [cell.text.strip() for cell in row.cells]
                        table_matrix.append(row_data)
                    md_table = table_to_markdown(table_matrix)
                    if md_table:
                        current_page_blocks.append(md_table)
                        char_count += len(md_table)
                        if char_count >= CHARS_PER_PAGE:
                            flush_page()
                    break

    flush_page()

    if not pages:
        pages.append(ExtractedPage(page_number=1, text="[Empty document]"))

    return ParsedDocument(
        filename=file_path.name,
        file_type="docx",
        total_pages=len(pages),
        pages=pages,
        metadata={"title": file_path.stem, "parser_used": "python-docx"}
    )


def extract_document(file_path: Path, preferred_parser: str = "docling") -> ParsedDocument:
    """
    Unified entry point for document extraction.
    Uses IBM Docling as the primary document understanding engine (preserving structure,
    hierarchy, tables, lists, and reading order).
    Falls back gracefully to PyMuPDF (PDF) or python-docx (DOCX) if Docling encounters an error.
    """
    suffix = file_path.suffix.lower()
    if suffix not in (".pdf", ".docx"):
        if suffix == ".doc":
            raise ValueError("Legacy .doc format is not directly supported. Please save or convert the file to .docx or .pdf before uploading.")
        raise ValueError(f"Unsupported file format '{suffix}'. Supported formats are: PDF (.pdf) and Word (.docx).")

    # Try Docling parser first if requested
    if preferred_parser == "docling":
        try:
            from Backend.docling_parser import parse_with_docling
            return parse_with_docling(file_path)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                f"Docling parsing encountered an issue for '{file_path.name}': {e}. Gracefully falling back to standard parser."
            )

    # Standard Fallback Parser (PyMuPDF / python-docx)
    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix == ".docx":
        return parse_docx(file_path)
