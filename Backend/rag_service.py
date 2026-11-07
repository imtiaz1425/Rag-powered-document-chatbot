import os
import re
import threading
from pathlib import Path
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv

from Backend.document_parser import extract_document, ParsedDocument
from Backend.chunking import create_chunks, DocumentChunk
from Backend.retriever import HybridRetriever

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_llm_instance():
    """
    Instantiate the primary Groq LLM client.
    Validates GROQ_API_KEY and provides clear configuration guidance if missing.
    """
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key or groq_key.strip() in ("", "your_groq_api_key_here"):
        raise ValueError(
            "GROQ_API_KEY is not configured. Please add your Groq API key to your local .env file:\n"
            "GROQ_API_KEY=gsk_your_actual_key_here\n"
            "You can get a free API key at: https://console.groq.com/keys"
        )

    from langchain_groq import ChatGroq
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    return ChatGroq(
        model=model,
        api_key=groq_key.strip(),
        temperature=0.1,
        max_retries=2,
        timeout=45.0
    )


class DocumentChatService:
    """
    Manages active document state, ingestion, hybrid retrieval, and grounded Q&A.
    Thread-safe to prevent race conditions during upload or inference.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self.active_document: Optional[ParsedDocument] = None
        self.chunks: List[DocumentChunk] = []
        self.retriever: Optional[HybridRetriever] = None
        self.session_info: Dict[str, Any] = {
            "status": "idle",
            "filename": None,
            "file_type": None,
            "total_pages": 0,
            "total_chunks": 0
        }

    def process_and_index_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Full ingestion pipeline:
        1. Extract text, tables, headings, and page boundaries.
        2. Create structure-aware, token-safe chunks.
        3. Build Hybrid FAISS + BM25 index.
        4. Atomically swap active document session.
        """
        with self._lock:
            # 1. Extraction
            parsed_doc = extract_document(file_path)
            
            # 2. Chunking (chunk_size=500, overlap=100)
            doc_chunks = create_chunks(parsed_doc, chunk_size=500, chunk_overlap=100)
            if not doc_chunks:
                raise ValueError("Could not extract any readable text or content from this document.")

            # 3. Hybrid Indexing (FAISS + BM25)
            new_retriever = HybridRetriever(doc_chunks)

            # 4. Atomic session update
            parser_engine = parsed_doc.metadata.get("parser_used", "docling")
            self.active_document = parsed_doc
            self.chunks = doc_chunks
            self.retriever = new_retriever
            self.session_info = {
                "status": "ready",
                "filename": parsed_doc.filename,
                "file_type": parsed_doc.file_type,
                "total_pages": parsed_doc.total_pages,
                "total_chunks": len(doc_chunks),
                "parser_used": parser_engine
            }

            return {
                "filename": parsed_doc.filename,
                "file_type": parsed_doc.file_type,
                "pages": parsed_doc.total_pages,
                "chunks": len(doc_chunks),
                "parser_used": parser_engine,
                "message": f"Your document is ready! I've processed your document and I'm ready to answer your questions. Ask me anything about it."
            }

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        Generate a factual, grounded response citing specific page numbers.
        """
        clean_q = question.strip()
        if not clean_q:
            return {"error": "Question cannot be empty."}

        with self._lock:
            if not self.active_document or not self.retriever:
                return {
                    "answer": "Please upload a document first before asking questions.",
                    "pages": [],
                    "source_document": None,
                    "retrieved_chunks": 0
                }

            current_doc = self.active_document
            retriever = self.retriever

        # Handle simple greetings without executing full retrieval
        greeting_words = {"hi", "hello", "hey", "who are you", "what can you do", "help"}
        if clean_q.lower() in greeting_words or (len(clean_q.split()) <= 2 and any(clean_q.lower().startswith(g) for g in greeting_words)):
            return {
                "answer": f"Hello! I am ready to answer any questions about your uploaded document **{current_doc.filename}** ({current_doc.total_pages} pages). What would you like to know?",
                "pages": [],
                "source_document": current_doc.filename,
                "retrieved_chunks": 0
            }

        # 1. Retrieve top-8 relevant chunks via Hybrid Retrieval
        retrieved_chunks = retriever.retrieve(clean_q, top_k=8)
        if not retrieved_chunks:
            return {
                "answer": "The document does not contain information to answer this question.",
                "pages": [],
                "source_document": current_doc.filename,
                "retrieved_chunks": 0
            }

        # 2. Construct context sorted in natural document page order
        sorted_chunks = sorted(
            retrieved_chunks,
            key=lambda c: (c.page_number, int(re.search(r"c(\d+)", c.chunk_id).group(1)) if re.search(r"c(\d+)", c.chunk_id) else 0)
        )
        context_blocks = []
        for i, chunk in enumerate(sorted_chunks, 1):
            context_blocks.append(
                f"--- [Page {chunk.page_number} | Section: {chunk.section_heading} | Item {i}] ---\n{chunk.content}"
            )
        context_str = "\n\n".join(context_blocks)

        # 3. Grounded Prompt
        prompt = f"""You are a helpful and precise Document Assistant.
Answer the user's question accurately using ONLY the provided document evidence below.

CRITICAL INSTRUCTIONS:
1. Ground every statement in the provided evidence. Do NOT invent, extrapolate, or assume facts not present in the text.
2. Provide page citations in your answer whenever referring to facts, figures, or claims, formatted strictly as: [Page X] (where X is the physical Page number from the header, e.g. [Page 4]). Never cite the Item number as the page.
3. Preserve exact numerical figures, currencies, dates, names, and metrics verbatim from the evidence. Never estimate or paraphrase numbers.
4. If the question asks for comparisons or multiple figures, present them clearly using markdown tables or bullet points.
5. If the provided evidence does not contain the answer, honestly respond:
"The provided document does not contain information to answer this question."
6. If the evidence only partially answers the question, answer what is supported with citations, and state clearly what is not mentioned.

DOCUMENT EVIDENCE:
{context_str}

USER QUESTION:
{clean_q}

YOUR GROUNDED ANSWER:"""

        try:
            llm = get_llm_instance()
            response = llm.invoke(prompt)
            answer_text = response.content if hasattr(response, "content") else str(response)
            if not answer_text or not answer_text.strip():
                return {
                    "answer": "Groq returned an empty response. Please rephrase your question or try again.",
                    "pages": [],
                    "source_document": current_doc.filename,
                    "retrieved_chunks": len(retrieved_chunks)
                }
        except ValueError as ve:
            # API key missing or placeholder
            return {
                "error": str(ve),
                "answer": f"Configuration Error:\n{str(ve)}",
                "pages": [],
                "source_document": current_doc.filename,
                "retrieved_chunks": len(retrieved_chunks)
            }
        except Exception as e:
            err_msg = str(e)
            if "rate_limit" in err_msg.lower() or "429" in err_msg:
                user_msg = "Groq API rate limit reached. Please wait a few seconds before submitting your next question."
            elif "model_not_found" in err_msg.lower() or "404" in err_msg:
                user_msg = f"Configured Groq model is currently unavailable on your account. Error: {err_msg}"
            else:
                user_msg = f"An error occurred while communicating with the Groq API: {err_msg}"

            return {
                "error": f"Groq Generation failed: {err_msg}",
                "answer": user_msg,
                "pages": [],
                "source_document": current_doc.filename,
                "retrieved_chunks": len(retrieved_chunks)
            }

        # Extract cited pages from answer text and retrieved chunks
        cited_pages = set()
        matches = re.findall(r"\[Page\s*(\d+)\]", answer_text, re.IGNORECASE)
        for m in matches:
            cited_pages.add(int(m))

        # Also collect pages from retrieved evidence
        evidence_pages = sorted(list(set(c.page_number for c in retrieved_chunks)))
        # If model didn't explicitly format [Page X], default to evidence pages
        final_pages = sorted(list(cited_pages)) if cited_pages else evidence_pages

        return {
            "answer": answer_text.strip(),
            "pages": final_pages,
            "source_document": current_doc.filename,
            "retrieved_chunks": len(retrieved_chunks),
            "evidence": [
                {
                    "page": c.page_number,
                    "section": c.section_heading,
                    "snippet": c.content[:160] + "..." if len(c.content) > 160 else c.content
                }
                for c in retrieved_chunks
            ]
        }

