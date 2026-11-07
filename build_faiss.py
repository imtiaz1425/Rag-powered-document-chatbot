from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


base_dir = Path(__file__).resolve().parent / "data" / "reports"
pdf_paths = sorted(base_dir.rglob("*.pdf"))

documents = []
for pdf_path in pdf_paths:
    documents.extend(PyPDFLoader(str(pdf_path)).load())

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings_model)

save_path = Path(__file__).resolve().parent / "vectorstore" / "faiss_index"
vectorstore.save_local(str(save_path))

print(f"total chunks indexed: {len(chunks)}")
print(f"embedding model name: sentence-transformers/all-MiniLM-L6-v2")
print(f"save location: {save_path}")
