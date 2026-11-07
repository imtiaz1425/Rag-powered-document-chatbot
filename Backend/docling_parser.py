import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel, DoclingDocument

from Backend.document_parser import ExtractedPage, ParsedDocument

logger = logging.getLogger(__name__)

# Base and cache directories
BASE_DIR = Path(__file__).resolve().parent.parent
DOCLING_CACHE_DIR = BASE_DIR / "data" / "docling_cache"
DOCLING_TEST_OUTPUT_DIR = BASE_DIR / "docling_test_output"
DOCLING_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for cache keying."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class DoclingDocumentParser:
    """
    High-fidelity document understanding parser powered by IBM Docling.
    Preserves:
    - Reading order
    - Hierarchy (Titles, Section Headers, Subheaders)
    - Tabular structure (Markdown tables with column alignment)
    - Lists, footnotes, captions, and text paragraphs
    - Page provenance / physical page boundaries
    """

    def __init__(self):
        self._converter: Optional[DocumentConverter] = None

    def _get_converter(self) -> DocumentConverter:
        """Lazy initialization of DocumentConverter to conserve memory until needed."""
        if self._converter is None:
            logger.info("Initializing IBM Docling DocumentConverter...")
            t0 = time.time()
            self._converter = DocumentConverter()
            logger.info(f"DocumentConverter initialized in {time.time() - t0:.2f}s")
        return self._converter

    def _check_cache(self, file_path: Path, file_hash: str) -> Optional[DoclingDocument]:
        """
        Check for an existing parsed DoclingDocument in cache or test output directory.
        Allows instant re-indexing without repeating expensive vision inference.
        """
        # 1. Check data/docling_cache/<sha256>.json
        cache_file = DOCLING_CACHE_DIR / f"{file_hash}.json"
        if cache_file.exists() and cache_file.stat().st_size > 0:
            try:
                logger.info(f"Loading Docling document from cache: {cache_file.name}")
                return DoclingDocument.load_from_json(cache_file)
            except Exception as e:
                logger.warning(f"Failed to load cached docling file {cache_file}: {e}")

        # 2. Check docling_test_output/<stem>_docling.json
        test_output_file = DOCLING_TEST_OUTPUT_DIR / f"{file_path.stem}_docling.json"
        if test_output_file.exists() and test_output_file.stat().st_size > 0:
            try:
                logger.info(f"Loading Docling document from test output: {test_output_file.name}")
                doc = DoclingDocument.load_from_json(test_output_file)
                # Populate cache for fast subsequent lookups
                try:
                    doc.save_as_json(cache_file)
                except Exception:
                    pass
                return doc
            except Exception as e:
                logger.warning(f"Failed to load test output docling file {test_output_file}: {e}")

        return None

    def parse_document(self, file_path: Path) -> ParsedDocument:
        """
        Parse a PDF or DOCX file using Docling, returning a standardized ParsedDocument.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found at: {file_path}")

        file_type = file_path.suffix.lower().lstrip(".")
        if file_type not in ("pdf", "docx"):
            raise ValueError(f"Docling parser only supports .pdf and .docx, got: {file_path.suffix}")

        t_start = time.time()
        file_hash = compute_file_sha256(file_path)
        cache_file = DOCLING_CACHE_DIR / f"{file_hash}.json"

        # Check cache first
        doc = self._check_cache(file_path, file_hash)
        is_cached = doc is not None

        if doc is None:
            converter = self._get_converter()
            logger.info(f"Executing Docling DocumentConverter on '{file_path.name}'...")
            conv_result = converter.convert(str(file_path))
            doc = conv_result.document
            # Cache the parsed document for future reuse
            try:
                doc.save_as_json(cache_file)
                logger.info(f"Saved parsed Docling document to cache: {cache_file.name}")
            except Exception as e:
                logger.warning(f"Could not save docling cache: {e}")

        parse_duration = time.time() - t_start
        total_pages = doc.num_pages()

        # Handle DOCX / documents where visual page provenance is not pre-rendered
        # For flow-based documents like Word, we create virtual pages (~2500 chars per page)
        is_flow_doc = (total_pages == 0)
        PAGE_CHAR_BUDGET = 2500
        current_flow_page = 1
        current_flow_chars = 0

        # Initialize page accumulator
        # If total_pages > 0, pre-seed pages 1 to total_pages to guarantee complete page sequence
        pages_data: Dict[int, Dict[str, Any]] = {}
        if not is_flow_doc:
            for p in range(1, total_pages + 1):
                pages_data[p] = {
                    "headings": [],
                    "tables": [],
                    "blocks": [],
                    "elements": []
                }
        else:
            pages_data[1] = {
                "headings": [],
                "tables": [],
                "blocks": [],
                "elements": []
            }

        # Iterate over all document items in reading order
        for item, level in doc.iterate_items():
            # Determine page number
            if not is_flow_doc:
                if hasattr(item, "prov") and item.prov and hasattr(item.prov[0], "page_no"):
                    p_num = item.prov[0].page_no
                else:
                    p_num = 1
                p_num = max(1, min(p_num, total_pages))
            else:
                p_num = current_flow_page

            if p_num not in pages_data:
                pages_data[p_num] = {
                    "headings": [],
                    "tables": [],
                    "blocks": [],
                    "elements": []
                }

            p_bucket = pages_data[p_num]
            label = item.label

            # 1. Section Headers & Titles
            if label in (DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER):
                text = getattr(item, "text", "").strip()
                if text:
                    h_level = max(1, min(level + 1, 4))
                    heading_prefix = "#" * h_level
                    md_heading = f"{heading_prefix} {text}"
                    p_bucket["headings"].append(text)
                    p_bucket["blocks"].append(md_heading)
                    p_bucket["elements"].append({
                        "type": "heading",
                        "level": h_level,
                        "text": text,
                        "markdown": md_heading
                    })
                    if is_flow_doc:
                        current_flow_chars += len(text)

            # 2. Tables
            elif label == DocItemLabel.TABLE:
                try:
                    table_md = item.export_to_markdown(doc=doc).strip()
                except Exception:
                    try:
                        table_md = item.export_to_markdown().strip()
                    except Exception:
                        table_md = ""

                if table_md:
                    caption = ""
                    if hasattr(item, "caption_text"):
                        try:
                            caption = item.caption_text(doc) or ""
                        except Exception:
                            caption = ""

                    p_bucket["tables"].append(table_md)
                    if caption:
                        p_bucket["blocks"].append(f"*Table: {caption}*\n\n{table_md}")
                    else:
                        p_bucket["blocks"].append(table_md)

                    p_bucket["elements"].append({
                        "type": "table",
                        "markdown": table_md,
                        "caption": caption
                    })
                    if is_flow_doc:
                        current_flow_chars += len(table_md)

            # 3. Lists
            elif label == DocItemLabel.LIST_ITEM:
                text = getattr(item, "text", "").strip()
                if text:
                    md_item = f"- {text}"
                    p_bucket["blocks"].append(md_item)
                    p_bucket["elements"].append({
                        "type": "list_item",
                        "text": text,
                        "markdown": md_item
                    })
                    if is_flow_doc:
                        current_flow_chars += len(text)

            # 4. Footnotes
            elif label == DocItemLabel.FOOTNOTE:
                text = getattr(item, "text", "").strip()
                if text:
                    md_note = f"[Footnote: {text}]"
                    p_bucket["blocks"].append(md_note)
                    p_bucket["elements"].append({
                        "type": "footnote",
                        "text": text,
                        "markdown": md_note
                    })
                    if is_flow_doc:
                        current_flow_chars += len(text)

            # 5. Captions
            elif label == DocItemLabel.CAPTION:
                text = getattr(item, "text", "").strip()
                if text:
                    p_bucket["blocks"].append(f"*Caption: {text}*")
                    p_bucket["elements"].append({
                        "type": "caption",
                        "text": text
                    })
                    if is_flow_doc:
                        current_flow_chars += len(text)

            # 6. Page Headers / Footers (filtered or preserved cleanly)
            elif label in (DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER):
                text = getattr(item, "text", "").strip()
                # Exclude trivial lone page numbers to avoid noise
                if text and not re.fullmatch(r"\d+", text):
                    p_bucket["elements"].append({
                        "type": "header_footer",
                        "text": text
                    })

            # 7. Standard Paragraphs / Text
            else:
                text = getattr(item, "text", "").strip()
                if text:
                    # Filter out purely decorative lines, underscores, dashes
                    if len(re.sub(r"[\s_—\-\.]+", "", text)) < 3:
                        continue
                    # Filter out lone signature placeholders that dilute table retrieval
                    if re.match(r"^(Managing Director|Director|Company Secretary|Auditor|Dhaka,\s*As per)", text, re.IGNORECASE):
                        continue
                    p_bucket["blocks"].append(text)
                    p_bucket["elements"].append({
                        "type": "paragraph",
                        "text": text
                    })
                    if is_flow_doc:
                        current_flow_chars += len(text)

            # Advance virtual page for flow documents when char budget is reached
            if is_flow_doc and current_flow_chars >= PAGE_CHAR_BUDGET:
                current_flow_page += 1
                current_flow_chars = 0
                if current_flow_page not in pages_data:
                    pages_data[current_flow_page] = {
                        "headings": [],
                        "tables": [],
                        "blocks": [],
                        "elements": []
                    }

        # Build final ExtractedPage list
        final_page_count = total_pages if not is_flow_doc else max(pages_data.keys(), default=1)
        extracted_pages: List[ExtractedPage] = []

        for p_idx in range(1, final_page_count + 1):
            p_dict = pages_data.get(p_idx, {"headings": [], "tables": [], "blocks": [], "elements": []})
            page_text = "\n\n".join(p_dict["blocks"]).strip()
            
            # Scanned page detection
            is_scanned = False
            if not page_text and not is_flow_doc:
                page_text = f"[Page {p_idx}: Empty or image-only content]"

            extracted_pages.append(ExtractedPage(
                page_number=p_idx,
                text=page_text,
                tables=p_dict["tables"],
                headings=p_dict["headings"],
                is_scanned=is_scanned,
                elements=p_dict["elements"]
            ))

        total_detected_tables = sum(len(p.tables) for p in extracted_pages)
        total_detected_headings = sum(len(p.headings) for p in extracted_pages)

        parsed_doc = ParsedDocument(
            filename=file_path.name,
            file_type=file_type,
            total_pages=final_page_count,
            pages=extracted_pages,
            metadata={
                "parser_used": "docling",
                "cached": is_cached,
                "parse_time_seconds": round(parse_duration, 2),
                "total_tables": total_detected_tables,
                "total_headings": total_detected_headings,
                "file_sha256": file_hash,
                "title": file_path.stem
            }
        )

        logger.info(
            f"Docling parsed '{file_path.name}': {final_page_count} pages, "
            f"{total_detected_tables} tables, {total_detected_headings} headings in {parse_duration:.2f}s "
            f"(cached={is_cached})"
        )

        return parsed_doc


# Global parser instance
_DOCLING_PARSER_INSTANCE: Optional[DoclingDocumentParser] = None


def get_docling_parser() -> DoclingDocumentParser:
    global _DOCLING_PARSER_INSTANCE
    if _DOCLING_PARSER_INSTANCE is None:
        _DOCLING_PARSER_INSTANCE = DoclingDocumentParser()
    return _DOCLING_PARSER_INSTANCE


def parse_with_docling(file_path: Path) -> ParsedDocument:
    """Convenience functional wrapper for Docling extraction."""
    return get_docling_parser().parse_document(file_path)
