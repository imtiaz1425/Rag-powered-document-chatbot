import sys
import time
from pathlib import Path
from Backend.document_parser import extract_document
from Backend.chunking import create_chunks
from Backend.retriever import HybridRetriever
from Backend.rag_service import DocumentChatService


def run_comprehensive_evaluation():
    print("=" * 70)
    print("GENERAL-PURPOSE AI DOCUMENT CHATBOT — COMPREHENSIVE PIPELINE AUDIT")
    print("=" * 70)

    service = DocumentChatService()
    results_summary = []

    # -------------------------------------------------------------
    # TEST 1: Word Document (.docx) Extraction, Tables, & Lists
    # -------------------------------------------------------------
    print("\n[TEST 1] Testing Word (.docx) Extraction & Structure Preservation...")
    docx_path = Path("data/test_docs/ai_ethics_guidelines.docx")
    parsed_docx = extract_document(docx_path)
    print(f"  Filename: {parsed_docx.filename}")
    print(f"  Pages: {parsed_docx.total_pages}")
    print(f"  Headings found: {parsed_docx.pages[0].headings}")
    
    # Check if table markdown is present in text
    has_table = "| Metric Category | Target Threshold | Audit Frequency |" in parsed_docx.pages[0].text
    has_bullets = "- Transparency and Explainability" in parsed_docx.pages[0].text
    print(f"  Table preserved as Markdown: {'PASS' if has_table else 'FAIL'}")
    print(f"  Bullet list preserved: {'PASS' if has_bullets else 'FAIL'}")
    results_summary.append(("DOCX Table & List Extraction", has_table and has_bullets))

    # -------------------------------------------------------------
    # TEST 2: Word Document Q&A & Numerical Grounding
    # -------------------------------------------------------------
    print("\n[TEST 2] Testing Word (.docx) Grounded Q&A...")
    service.process_and_index_document(docx_path)

    # Q2.1: Specific tabular figure
    q_table = "What is the target threshold and audit frequency for Model Explainability Score?"
    ans_table = service.answer_question(q_table)
    print(f"  Q: {q_table}")
    print(f"  A: {ans_table['answer']}")
    print(f"  Pages Cited: {ans_table['pages']}")
    ans_lower = ans_table["answer"].lower()
    q1_passed = "85" in ans_lower and "quarterly" in ans_lower
    print(f"  Verification (85% & Quarterly): {'PASS' if q1_passed else 'FAIL'}")
    results_summary.append(("DOCX Tabular Fact Retrieval", q1_passed))

    # Q2.2: Unsupported question fallback
    q_unsupported = "What is the penalty for quantum computing export violations?"
    ans_unsupported = service.answer_question(q_unsupported)
    print(f"\n  Q (Unsupported): {q_unsupported}")
    print(f"  A: {ans_unsupported['answer']}")
    is_honest_fallback = "not contain" in ans_unsupported["answer"].lower() or "does not state" in ans_unsupported["answer"].lower()
    print(f"  Honest Fallback: {'PASS' if is_honest_fallback else 'FAIL'}")
    results_summary.append(("Honest Unsupported Fallback", is_honest_fallback))

    # -------------------------------------------------------------
    # TEST 3: Document Session Replacement
    # -------------------------------------------------------------
    print("\n[TEST 3] Testing Document Session Replacement (Replacing DOCX with PDF)...")
    pdf_path = Path("data/uploads/bank of america 10kreports.pdf")
    t0 = time.time()
    pdf_index_res = service.process_and_index_document(pdf_path)
    index_time = time.time() - t0
    print(f"  Indexed {pdf_index_res['filename']}: {pdf_index_res['pages']} pages, {pdf_index_res['chunks']} chunks in {index_time:.2f}s")
    
    # Verify active session changed
    session_clean = service.session_info["filename"] == "bank of america 10kreports.pdf"
    print(f"  Session replaced cleanly: {'PASS' if session_clean else 'FAIL'}")
    results_summary.append(("Session Replacement Integrity", session_clean))

    # -------------------------------------------------------------
    # TEST 4: Multi-Page PDF Specific Fact Retrieval & Page Citations
    # -------------------------------------------------------------
    print("\n[TEST 4] Testing Multi-Page PDF Retrieval & Citations...")
    q_pdf = "What does the filing say about the exchange where securities are registered?"
    ans_pdf = service.answer_question(q_pdf)
    print(f"  Q: {q_pdf}")
    print(f"  A: {ans_pdf['answer']}")
    print(f"  Pages Cited: {ans_pdf['pages']}")
    pdf_passed = len(ans_pdf["pages"]) > 0 and ("n/a" in ans_pdf["answer"].lower() or "exchange" in ans_pdf["answer"].lower())
    print(f"  Page Citations Present & Factual: {'PASS' if pdf_passed else 'FAIL'}")
    results_summary.append(("Multi-Page Citations & Retrieval", pdf_passed))

    # -------------------------------------------------------------
    # TEST 5: Long PDF (100+ pages) Performance & Speed
    # -------------------------------------------------------------
    print("\n[TEST 5] Testing Long PDF (100+ pages) Extraction Speed & Scalability...")
    long_pdf_path = Path("data/reports/c87043b9-5d89-4717-9f49-c4f9663d0061.pdf")
    t0 = time.time()
    parsed_long = extract_document(long_pdf_path)
    extract_time = time.time() - t0
    print(f"  Extracted {parsed_long.total_pages} pages in {extract_time:.2f}s (Speed: {parsed_long.total_pages / max(extract_time, 0.001):.1f} pages/sec)")
    long_pass = parsed_long.total_pages >= 100 and extract_time < 5.0
    print(f"  Long Document High-Speed Extraction: {'PASS' if long_pass else 'FAIL'}")
    results_summary.append(("Long PDF (100+ pages) Extraction", long_pass))

    # -------------------------------------------------------------
    # TEST 6: Scanned Page Detection
    # -------------------------------------------------------------
    print("\n[TEST 6] Testing Scanned / Image-Based Page Detection...")
    scanned_pdf_path = Path("data/uploads/United Group_Annual-Report-2022-23.pdf")
    # United Group page 1 (cover) has 0 text
    ug_doc = extract_document(scanned_pdf_path)
    scanned_detected = ug_doc.pages[0].is_scanned
    print(f"  Cover page is_scanned flag: {scanned_detected}")
    print(f"  Cover page text snippet: {repr(ug_doc.pages[0].text[:80])}")
    results_summary.append(("Scanned Page Detection", scanned_detected))

    # -------------------------------------------------------------
    # Summary Table
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    all_pass = True
    for name, passed in results_summary:
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  - {name:38}: [{status_str}]")
    print("=" * 70)
    print(f"OVERALL EVALUATION: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}\n")


if __name__ == "__main__":
    run_comprehensive_evaluation()
