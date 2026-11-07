import re
from typing import Any, Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Backend.document_parser import ParsedDocument, ExtractedPage


class DocumentChunk:
    def __init__(
        self,
        chunk_id: str,
        content: str,
        page_number: int,
        document_name: str,
        section_heading: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.content = content.strip()
        self.page_number = page_number
        self.document_name = document_name
        self.section_heading = section_heading or "General"
        self.metadata = metadata or {}

    def get_embedding_text(self) -> str:
        """
        Text used for generating vector embeddings.
        Prefixed with page and section context to maximize retrieval accuracy.
        """
        prefix = f"[{self.document_name} | Page {self.page_number}"
        if self.section_heading and self.section_heading != "General":
            prefix += f" | {self.section_heading}"
        prefix += "]\n"
        return prefix + self.content

    def get_context_text(self) -> str:
        """Formatted text passed to the LLM during generation with explicit citations."""
        return f"[Document: {self.document_name} | Page {self.page_number} | Section: {self.section_heading}]\n{self.content}"


def is_markdown_table(text: str) -> bool:
    """Check if a text block represents a formatted Markdown table."""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if len(lines) >= 2 and "|" in lines[0] and ("|" in lines[1] and "---" in lines[1]):
        return True
    return False


def split_markdown_table(table_md: str, max_chars: int = 1200) -> List[str]:
    """
    Carefully split a large Markdown table across rows while repeating the table header.
    Guarantees column semantics, comparative years, and headers are never lost across chunks,
    preventing silent truncation by embedding models with tight token budgets (e.g. 256 tokens).
    """
    lines = [l.strip() for l in table_md.strip().splitlines() if l.strip()]
    if len(lines) <= 2:
        return [table_md]

    header_line = lines[0]
    separator_line = lines[1]
    data_rows = lines[2:]

    # Keep small tables completely intact
    if len(table_md) <= max_chars or len(data_rows) <= 3:
        return [table_md]

    chunks: List[str] = []
    current_rows: List[str] = []
    header_len = len(header_line) + len(separator_line) + 2
    current_len = header_len

    for row in data_rows:
        row_len = len(row) + 1
        # If adding this row exceeds max_chars and we already have at least 1 row, flush current batch
        if current_rows and (current_len + row_len > max_chars):
            chunks.append("\n".join([header_line, separator_line] + current_rows))
            current_rows = [row]
            current_len = header_len + row_len
        else:
            current_rows.append(row)
            current_len += row_len

    if current_rows:
        chunks.append("\n".join([header_line, separator_line] + current_rows))

    return chunks


def create_chunks(
    parsed_doc: ParsedDocument,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[DocumentChunk]:
    """
    Split a ParsedDocument into structure-aware, token-safe chunks.
    - Preserves table rows and repeats table headers for large tables.
    - Tracks active headings across pages and blocks.
    - Respects token boundaries (~500 chars / ~100-130 tokens) to prevent silent truncation.
    - Preserves physical page numbers for exact citations.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
    )

    all_chunks: List[DocumentChunk] = []
    chunk_counter = 0
    current_active_heading = "General"

    for page in parsed_doc.pages:
        page_num = page.page_number
        raw_text = page.text
        if not raw_text.strip():
            continue

        # If page has declared headings, use the first one if we don't have an active heading
        if page.headings:
            current_active_heading = page.headings[0]

        blocks = raw_text.split("\n\n")

        for b in blocks:
            b_stripped = b.strip()
            if not b_stripped:
                continue

            # Check if block is a markdown heading (e.g. # Title, ## Section, ### Sub)
            heading_match = re.match(r"^#{1,6}\s+(.+)$", b_stripped)
            if heading_match and len(b_stripped.splitlines()) == 1:
                current_active_heading = heading_match.group(1).strip()
                continue

            # 1. Table Handling
            if is_markdown_table(b_stripped):
                table_sub_chunks = split_markdown_table(b_stripped, max_chars=1200)
                for sub_t in table_sub_chunks:
                    chunk_id = f"p{page_num}_c{chunk_counter}"
                    chunk_counter += 1
                    all_chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        content=sub_t,
                        page_number=page_num,
                        document_name=parsed_doc.filename,
                        section_heading=current_active_heading,
                        metadata={
                            "source": parsed_doc.filename,
                            "page": page_num,
                            "chunk_id": chunk_id,
                            "is_table": True,
                            "is_scanned": page.is_scanned
                        }
                    ))

            # 2. Text / Paragraph / List Handling
            else:
                if len(b_stripped) <= chunk_size:
                    sub_texts = [b_stripped]
                else:
                    sub_texts = splitter.split_text(b_stripped)

                for sub_text in sub_texts:
                    clean_sub = sub_text.strip()
                    if not clean_sub or len(clean_sub) < 15:
                        continue
                    # Filter out decorative underscore/dash lines
                    if len(re.sub(r"[\s_—\-\.]+", "", clean_sub)) < 5:
                        continue
                    chunk_id = f"p{page_num}_c{chunk_counter}"
                    chunk_counter += 1
                    all_chunks.append(DocumentChunk(
                        chunk_id=chunk_id,
                        content=clean_sub,
                        page_number=page_num,
                        document_name=parsed_doc.filename,
                        section_heading=current_active_heading,
                        metadata={
                            "source": parsed_doc.filename,
                            "page": page_num,
                            "chunk_id": chunk_id,
                            "is_table": False,
                            "is_scanned": page.is_scanned
                        }
                    ))

    return all_chunks
