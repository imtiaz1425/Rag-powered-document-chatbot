import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from Backend.chunking import DocumentChunk

# Global shared embeddings model to avoid reloading on each upload
_EMBEDDINGS_MODEL = None


def get_embeddings_model():
    global _EMBEDDINGS_MODEL
    if _EMBEDDINGS_MODEL is None:
        _EMBEDDINGS_MODEL = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True}  # Normalizes for true Cosine Similarity
        )
    return _EMBEDDINGS_MODEL


def tokenize_text(text: str) -> List[str]:
    """Simple alphanumeric tokenizer for BM25 keyword matching."""
    return [t.lower() for t in re.findall(r"\b\w+\b", text) if len(t) > 1]


def expand_query(query: str) -> List[str]:
    """
    Deterministic query expansion for complex, multi-entity, or conversational questions.
    Generates targeted sub-queries without extra LLM/API calls.
    """
    queries = [query]
    
    # 1. Clean conversational framing, instructions, and punctuation
    cleaned = re.sub(
        r'(?i)\b(also\s+show\s+the\s+calculation\s+using|show\s+the\s+calculation\s+using|show\s+the\s+calculation|and\s+cite\s+the\s+source\s+page|cite\s+the\s+source\s+page|what\s+(was|is|were|are)\s+the|please\s+(show|find|provide|tell\s+me)|can\s+you\s+(tell|show|find|provide))\b',
        ' ',
        query
    )
    cleaned = re.sub(r'[\?,\.]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    if cleaned and cleaned.lower() != query.lower():
        queries.append(cleaned)

    # 2. Extract 4-digit years
    years = re.findall(r'\b(19\d\d|20\d\d)\b', query)
    year_str = ' '.join(years) if years else ''

    q_lower = query.lower()
    has_assets = 'asset' in q_lower
    has_equity = 'equity' in q_lower
    has_liabilities = 'liabilit' in q_lower
    has_calc_or_bal = any(w in q_lower for w in ['balance sheet', 'statement of financial position', 'calculation', 'total assets'])

    # 3. Targeted sub-queries for balance sheet / statement of financial position
    if has_assets:
        if year_str:
            queries.append(f"Statement of financial position total assets {year_str}")
        queries.append("Statement of financial position total assets")
    if has_equity:
        if year_str:
            queries.append(f"Statement of financial position total equity {year_str}")
        queries.append("Statement of financial position total equity")
    if has_liabilities:
        if year_str:
            queries.append(f"Statement of financial position total liabilities {year_str}")
        queries.append("Statement of financial position total liabilities")
    if (has_assets or has_equity or has_liabilities) and (has_calc_or_bal or len(years) > 1):
        if year_str:
            queries.append(f"Statement of financial position balance sheet {year_str} assets equity liabilities")
        queries.append("Statement of financial position balance sheet")

    # 4. Income statement / profit or loss
    has_profit = any(w in q_lower for w in ['profit', 'revenue', 'turnover', 'income', 'expense'])
    if has_profit and any(w in q_lower for w in ['statement', 'income', 'comprehensive', 'loss']):
        if year_str:
            queries.append(f"Statement of profit or loss comprehensive income {year_str} revenue profit")
        queries.append("Statement of profit or loss comprehensive income")

    # 5. Cash flows
    has_cash = 'cash' in q_lower and ('flow' in q_lower or 'operating' in q_lower or 'activities' in q_lower)
    if has_cash:
        if year_str:
            queries.append(f"Statement of cash flows {year_str} operating investing financing")
        queries.append("Statement of cash flows")

    # Deduplicate preserving original query order
    seen = set()
    result = []
    for q in queries:
        q_norm = q.lower().strip()
        if q_norm and q_norm not in seen:
            seen.add(q_norm)
            result.append(q)
    return result


class CrossEncoderReranker:
    """
    Lightweight, modular cross-encoder reranker.
    Configured to strictly use storage on Drive D: to avoid exhausting Drive C: disk space.
    """
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_folder: Optional[str] = None
    ):
        self.model_name = model_name
        self.cache_folder = cache_folder or str(Path(__file__).resolve().parent.parent / "data" / "hf_cache")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            Path(self.cache_folder).mkdir(parents=True, exist_ok=True)
            self._model = CrossEncoder(self.model_name, cache_folder=self.cache_folder)
        return self._model

    def rerank(self, query: str, candidates: List[DocumentChunk], top_k: int = 8) -> List[DocumentChunk]:
        if not candidates or len(candidates) <= top_k:
            return candidates[:top_k]
        pairs = [[query, c.content] for c in candidates]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:top_k]]


class HybridRetriever:
    """
    Advanced Hybrid Retriever combining:
    1. Multi-query deterministic expansion.
    2. Dense FAISS semantic search with table diversity capping.
    3. Sparse BM25 lexical search with table diversity capping.
    4. Reciprocal Rank Fusion (RRF, k=60) with quantitative table weighting.
    5. Section/page balance filtering.
    6. Modular, optional cross-encoder reranking.
    """
    def __init__(
        self,
        chunks: List[DocumentChunk],
        fetch_k: int = 60,
        reranker: Optional[CrossEncoderReranker] = None,
        use_reranker: bool = False
    ):
        self.chunks = chunks
        self.chunk_map: Dict[str, DocumentChunk] = {c.chunk_id: c for c in chunks}
        self.fetch_k = fetch_k
        self.reranker = reranker
        self.use_reranker = use_reranker or (os.getenv("ENABLE_RERANKER", "false").lower() == "true")
        
        # 1. Build FAISS index
        embeddings = get_embeddings_model()
        langchain_docs = [
            Document(
                page_content=c.get_embedding_text(),
                metadata={"chunk_id": c.chunk_id, "page": c.page_number}
            )
            for c in chunks
        ]
        self.faiss_store = FAISS.from_documents(langchain_docs, embeddings)

        # 2. Build BM25 index
        corpus_tokens = [tokenize_text(c.get_embedding_text()) for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)

    def retrieve(self, query: str, top_k: int = 8) -> List[DocumentChunk]:
        """
        Execute multi-query hybrid retrieval with table-aware candidate selection,
        reciprocal rank fusion, and diversity balancing.
        """
        if not self.chunks:
            return []

        # 1. Expand query into focused sub-queries
        sub_queries = expand_query(query)
        effective_fetch_k = min(len(self.chunks), max(self.fetch_k, top_k * 4))

        # 2. Identify if query targets tables, quantitative values, or statements
        q_lower = query.lower()
        is_table_query = any(w in q_lower for w in [
            'total', 'amount', 'calculate', 'calculation', 'table', 'balance sheet',
            'statement', 'equity', 'asset', 'liabilit', 'revenue', 'profit', 'cash'
        ])

        # 3. Detect explicit page number request (e.g. "page 4", "page 42")
        target_page = None
        page_match = re.search(r"\bpage\s*(\d+)\b", query, re.IGNORECASE)
        if page_match:
            try:
                target_page = int(page_match.group(1))
            except ValueError:
                pass

        rrf_scores: Dict[str, float] = {}

        # 4. Multi-Query Search & Reciprocal Rank Fusion
        for sq in sub_queries:
            # 4a. Dense FAISS search with table candidate diversity capping
            dense_raw = self.faiss_store.similarity_search_with_score(sq, k=effective_fetch_k * 2)
            dense_candidates = []
            tbl_dense_counts: Dict[str, int] = {}
            for doc, _ in dense_raw:
                cid = doc.metadata.get("chunk_id")
                c = self.chunk_map[cid]
                is_tbl = c.metadata.get("is_table", False)
                tbl_key = f"p{c.page_number}_{c.section_heading}" if is_tbl else None
                if tbl_key:
                    tbl_dense_counts[tbl_key] = tbl_dense_counts.get(tbl_key, 0) + 1
                    # Avoid allowing many nearly identical chunks from one large table to monopolize candidate pool
                    if tbl_dense_counts[tbl_key] > 4:
                        continue
                dense_candidates.append(cid)
                if len(dense_candidates) >= effective_fetch_k:
                    break

            # 4b. Sparse BM25 search with table candidate diversity capping
            q_toks = tokenize_text(sq)
            if q_toks:
                bm_scores = self.bm25.get_scores(q_toks)
                ranked_bm_indices = sorted(range(len(bm_scores)), key=lambda i: bm_scores[i], reverse=True)
                bm_candidates = []
                tbl_bm_counts: Dict[str, int] = {}
                for idx in ranked_bm_indices:
                    c = self.chunks[idx]
                    cid = c.chunk_id
                    is_tbl = c.metadata.get("is_table", False)
                    tbl_key = f"p{c.page_number}_{c.section_heading}" if is_tbl else None
                    if tbl_key:
                        tbl_bm_counts[tbl_key] = tbl_bm_counts.get(tbl_key, 0) + 1
                        if tbl_bm_counts[tbl_key] > 4:
                            continue
                    bm_candidates.append(cid)
                    if len(bm_candidates) >= effective_fetch_k:
                        break
            else:
                bm_candidates = []

            # 4c. RRF Accumulation (k=60)
            for rank, cid in enumerate(dense_candidates):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank + 1))
            for rank, cid in enumerate(bm_candidates):
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (60.0 + rank + 1))

        # 5. Table & Quantitative weighting
        if is_table_query:
            for cid in list(rrf_scores.keys()):
                if self.chunk_map[cid].metadata.get("is_table"):
                    rrf_scores[cid] *= 1.8

        # 6. Explicit page target boost
        if target_page is not None:
            for cid in list(rrf_scores.keys()):
                if self.chunk_map[cid].page_number == target_page:
                    rrf_scores[cid] += 0.2

        # 7. Sort all unique candidates by fused score
        sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        # 8. Diversity-aware selection (prevent single page monopolizing unless explicit target)
        selected_chunk_ids: List[str] = []
        page_counts: Dict[int, int] = {}
        max_per_page = top_k if (target_page is not None) else 3

        for cid in sorted_chunk_ids:
            chunk = self.chunk_map[cid]
            p = chunk.page_number
            if target_page is not None and p == target_page:
                selected_chunk_ids.append(cid)
            elif page_counts.get(p, 0) < max_per_page:
                selected_chunk_ids.append(cid)
                page_counts[p] = page_counts.get(p, 0) + 1
            if len(selected_chunk_ids) >= (top_k * 2 if self.use_reranker else top_k):
                break

        # Fallback if diversity was too strict
        if len(selected_chunk_ids) < min(top_k, len(self.chunks)):
            for cid in sorted_chunk_ids:
                if cid not in selected_chunk_ids:
                    selected_chunk_ids.append(cid)
                if len(selected_chunk_ids) >= top_k:
                    break

        candidate_chunks = [self.chunk_map[cid] for cid in selected_chunk_ids]

        # 9. Optional Cross-Encoder Reranking
        if self.use_reranker:
            reranker = self.reranker or CrossEncoderReranker()
            return reranker.rerank(query, candidate_chunks, top_k=top_k)

        return candidate_chunks[:top_k]
