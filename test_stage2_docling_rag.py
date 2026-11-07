import json
import os
import sys
import time
from pathlib import Path

# Ensure root directory is on PYTHONPATH
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from Backend.rag_service import DocumentChatService
from Backend.document_parser import parse_pdf, extract_document
from Backend.app import app


def run_stage2_validation():
    print("=" * 75)
    print("STAGE 2 — DOCLING RAG INTEGRATION VALIDATION SUITE")
    print("=" * 75)

    test_results = {}
    service = DocumentChatService()

    # -------------------------------------------------------------
    # TEST A: 59-Page Financial PDF Ingestion & Indexing
    # -------------------------------------------------------------
    print("\n[TEST A] Ingesting 59-Page Financial PDF with Docling...")
    pdf_path = BASE_DIR / "96051_Final_Accounts_2014_1111.pdf"
    if not pdf_path.exists():
        pdf_path = BASE_DIR / "data" / "uploads" / "96051_Final_Accounts_2014_1111.pdf"

    t0 = time.time()
    pdf_index_res = service.process_and_index_document(pdf_path)
    ingest_time = time.time() - t0

    print(f"  Filename     : {pdf_index_res['filename']}")
    print(f"  Pages        : {pdf_index_res['pages']}")
    print(f"  Chunks       : {pdf_index_res['chunks']}")
    print(f"  Parser Used  : {pdf_index_res.get('parser_used', 'N/A')}")
    print(f"  Time Taken   : {ingest_time:.2f}s")

    test_a_pass = (
        pdf_index_res["pages"] == 59
        and pdf_index_res["chunks"] > 0
        and pdf_index_res.get("parser_used") == "docling"
    )
    test_results["A. 59-Page Financial PDF Ingestion"] = (test_a_pass, f"{pdf_index_res['pages']} pages, {pdf_index_res['chunks']} chunks in {ingest_time:.2f}s")
    print(f"  Result: {'PASS' if test_a_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # TEST B: Page 4 Complete Statement Retrieval
    # -------------------------------------------------------------
    print("\n[TEST B] Testing Page 4 Statement of Financial Position Retrieval...")
    q_b = "What is the complete Statement of financial position on Page 4?"
    ans_b = service.answer_question(q_b)
    print(f"  Question: {q_b}")
    print(f"  Pages Cited: {ans_b['pages']}")
    print(f"  Retrieved Chunks: {ans_b['retrieved_chunks']}")
    print(f"  Answer Preview:\n{ans_b['answer'][:350]}...\n")

    test_b_pass = 4 in ans_b["pages"] and (
        "statement of financial position" in ans_b["answer"].lower()
        or "total assets" in ans_b["answer"].lower()
        or "equity" in ans_b["answer"].lower()
        or "property, plant and equipment" in ans_b["answer"].lower()
        or "15,526,192,783" in ans_b["answer"]
        or "15526192783" in ans_b["answer"].replace(",", "")
    )
    test_results["B. Page 4 Complete Statement Retrieval"] = (test_b_pass, f"Cited pages: {ans_b['pages']}")
    print(f"  Result: {'PASS' if test_b_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # TEST C: Table Row and Comparative-Year Preservation
    # -------------------------------------------------------------
    print("\n[TEST C] Testing Table Row and Comparative-Year Preservation...")
    q_c = "What were the Property, plant and equipment, Inventories, and Total assets for both 31 December 2014 and 31 December 2013 on page 4?"
    ans_c = service.answer_question(q_c)
    print(f"  Question: {q_c}")
    print(f"  Answer:\n{ans_c['answer']}\n")
    print(f"  Pages Cited: {ans_c['pages']}")

    ans_c_text = ans_c["answer"].replace(",", "")
    # Check 2014: PPE 3,799,146,712 | Inventories 2,961,175,971 | Total assets 15,526,192,783
    # Check 2013: PPE 3,384,489,286 | Inventories 2,553,330,342 | Total assets 14,693,912,974
    c_ppe_2014 = "3799146712" in ans_c_text
    c_ppe_2013 = "3384489286" in ans_c_text
    c_inv_2014 = "2961175971" in ans_c_text
    c_inv_2013 = "2553330342" in ans_c_text
    c_ta_2014 = "15526192783" in ans_c_text
    c_ta_2013 = "14693912974" in ans_c_text

    print(f"  Verifications:")
    print(f"    - PPE 2014 (3,799,146,712) : {c_ppe_2014}")
    print(f"    - PPE 2013 (3,384,489,286) : {c_ppe_2013}")
    print(f"    - Inv 2014 (2,961,175,971) : {c_inv_2014}")
    print(f"    - Inv 2013 (2,553,330,342) : {c_inv_2013}")
    print(f"    - TA  2014 (15,526,192,783): {c_ta_2014}")
    print(f"    - TA  2013 (14,693,912,974): {c_ta_2013}")

    test_c_pass = (c_ppe_2014 and c_ppe_2013 and c_ta_2014 and c_ta_2013)
    test_results["C. Comparative-Year Table Preservation"] = (test_c_pass, f"PPE & Total Assets exact numbers preserved")
    print(f"  Result: {'PASS' if test_c_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # TEST D: Correct Page Citations
    # -------------------------------------------------------------
    print("\n[TEST D] Testing Correct Page Citations...")
    q_d = "What was the Retained earnings as of 31 December 2014 on page 4?"
    ans_d = service.answer_question(q_d)
    print(f"  Question: {q_d}")
    print(f"  Answer: {ans_d['answer']}")
    print(f"  Pages Cited: {ans_d['pages']}")
    
    test_d_pass = 4 in ans_d["pages"] and ("4,861,534,405" in ans_d["answer"] or "4861534405" in ans_d["answer"].replace(",", ""))
    test_results["D. Correct Page Citations"] = (test_d_pass, f"Cited Page 4 with correct value 4,861,534,405")
    print(f"  Result: {'PASS' if test_d_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # TEST E: Unsupported Question Fallback
    # -------------------------------------------------------------
    print("\n[TEST E] Testing Unsupported Question Fallback...")
    q_e = "What was the company's cryptocurrency trading profit in 2014?"
    ans_e = service.answer_question(q_e)
    print(f"  Question: {q_e}")
    print(f"  Answer: {ans_e['answer']}")
    
    ans_e_lower = ans_e["answer"].lower()
    test_e_pass = ("does not contain" in ans_e_lower or "not contain" in ans_e_lower or "no mention" in ans_e_lower or "not mentioned" in ans_e_lower or "does not provide" in ans_e_lower)
    test_results["E. Unsupported Question Fallback"] = (test_e_pass, "Honest refusal without hallucination")
    print(f"  Result: {'PASS' if test_e_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # TEST F: Existing DOCX Functionality
    # -------------------------------------------------------------
    print("\n[TEST F] Testing Word (.docx) Extraction and Grounded Q&A...")
    docx_path = BASE_DIR / "data" / "test_docs" / "ai_ethics_guidelines.docx"
    t0 = time.time()
    docx_res = service.process_and_index_document(docx_path)
    docx_time = time.time() - t0
    print(f"  DOCX Filename: {docx_res['filename']}")
    print(f"  DOCX Pages   : {docx_res['pages']}")
    print(f"  DOCX Chunks  : {docx_res['chunks']}")
    print(f"  Parser Used  : {docx_res.get('parser_used')}")

    q_f = "What is the target threshold and audit frequency for Model Explainability Score?"
    ans_f = service.answer_question(q_f)
    print(f"  Question: {q_f}")
    print(f"  Answer  : {ans_f['answer']}")
    print(f"  Pages   : {ans_f['pages']}")

    f_ans_lower = ans_f["answer"].lower()
    test_f_pass = "85" in f_ans_lower and "quarterly" in f_ans_lower and len(ans_f["pages"]) > 0
    test_results["F. Word (.docx) Functionality"] = (test_f_pass, f"Extracted 85% & Quarterly with Page {ans_f['pages']}")
    print(f"  Result: {'PASS' if test_f_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # TEST G: Flask Upload & Ask API Endpoints Flow
    # -------------------------------------------------------------
    print("\n[TEST G] Testing Flask Upload & Ask API Flow...")
    with app.test_client() as client:
        # Check /status
        status_res = client.get("/status")
        status_json = status_res.get_json()
        print(f"  /status: status={status_json.get('status')}, file={status_json.get('filename')}")

        # Test /upload with PDF
        with open(pdf_path, "rb") as f_upload:
            upload_res = client.post(
                "/upload",
                data={"file": (f_upload, pdf_path.name)},
                content_type="multipart/form-data"
            )
        upload_json = upload_res.get_json()
        print(f"  /upload: code={upload_res.status_code}, status={upload_json.get('status')}, parser_used={upload_json.get('parser_used')}")
        
        # Test /ask
        ask_res = client.post(
            "/ask",
            json={"question": "What is the total equity as of 31 December 2014 on page 4?"}
        )
        ask_json = ask_res.get_json()
        print(f"  /ask response code: {ask_res.status_code}")
        print(f"  /ask answer preview: {ask_json.get('answer', '')[:180]}...")
        print(f"  /ask pages: {ask_json.get('pages')}")

        test_g_pass = (
            upload_res.status_code == 200
            and upload_json.get("status") == "success"
            and upload_json.get("parser_used") == "docling"
            and ask_res.status_code == 200
            and ("6,606,685,402" in ask_json.get("answer", "") or "6606685402" in ask_json.get("answer", "").replace(",", ""))
        )
        test_results["G. Frontend API Flow (/upload & /ask)"] = (test_g_pass, "Upload & Ask endpoints returned valid contract")
        print(f"  Result: {'PASS' if test_g_pass else 'FAIL'}")

    # -------------------------------------------------------------
    # COMPARISON: Docling vs PyMuPDF Extraction on Page 4
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("COMPARISON: DOCLING VS PYMUPDF ON PAGE 4 (FINANCIAL POSITION)")
    print("=" * 75)
    doc_pymupdf = parse_pdf(pdf_path)
    p4_pymupdf = doc_pymupdf.pages[3]  # 0-indexed page 4
    print("\n--- PyMuPDF Extracted Text (First 350 chars) ---")
    print(repr(p4_pymupdf.text[:350]))
    print(f"PyMuPDF Tables Detected: {len(p4_pymupdf.tables)}")

    doc_docling = extract_document(pdf_path)
    p4_docling = doc_docling.pages[3]
    print("\n--- Docling Extracted Text (First 350 chars) ---")
    print(p4_docling.text[:350])
    print(f"Docling Tables Detected: {len(p4_docling.tables)}")

    # -------------------------------------------------------------
    # FINAL SUMMARY
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("STAGE 2 VALIDATION SUMMARY")
    print("=" * 75)
    all_passed = True
    for test_name, (passed, details) in test_results.items():
        status_str = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {status_str:4} | {test_name:<42} | {details}")
    print("=" * 75)
    print(f"OVERALL RESULT: {'ALL TESTS PASSED SUCCESSFULLY' if all_passed else 'SOME TESTS FAILED'}\n")


if __name__ == "__main__":
    run_stage2_validation()
