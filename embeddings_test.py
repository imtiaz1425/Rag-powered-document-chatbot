from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


base_dir = Path(__file__).resolve().parent / "data" / "reports"
pdf_paths = sorted(base_dir.rglob("*.pdf"))

documents = []
for pdf_path in pdf_paths:
    documents.extend(PyPDFLoader(str(pdf_path)).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
texts = [chunk.page_content for chunk in chunks]

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embeddings_model.embed_documents(texts) if texts else []
embedding_dimension = len(embeddings[0]) if embeddings else 0

print(f"total chunks: {len(chunks)}")
print(f"embedding dimension: {embedding_dimension}")
print(f"total embeddings created: {len(embeddings)}")
print(f"sample embedding length: {len(embeddings[0])}")
print(f"first 5 values: {embeddings[0][:5]}")
