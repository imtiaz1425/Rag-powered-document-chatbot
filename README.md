# RAG-Powered Document Chatbot

An AI-powered document question-answering system that allows users to upload PDF and DOCX files and ask questions using Retrieval-Augmented Generation (RAG).

The system combines document parsing, structure-aware chunking, hybrid retrieval, and Large Language Models to generate context-aware answers from uploaded documents.

## Key Features

- PDF and DOCX document processing
- Document extraction using Docling
- Structure-aware text chunking
- Semantic search using embeddings
- FAISS vector database
- BM25 keyword-based retrieval
- Hybrid retrieval combining semantic and keyword search
- AI-generated responses using Groq
- Interactive web-based chatbot interface

## Technology Stack

| Component | Technology |
|---|---|
| Backend | Python, Flask |
| Frontend | HTML, CSS, JavaScript |
| Document Processing | Docling |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| Keyword Retrieval | BM25 |
| LLM Provider | Groq |
| Retrieval Architecture | Hybrid RAG |

## System Architecture

1. **Document Upload:** Users upload PDF or DOCX documents.
2. **Document Extraction:** Docling extracts document content.
3. **Chunking:** Extracted content is divided into meaningful chunks.
4. **Embedding Generation:** Text chunks are converted into vector embeddings.
5. **Indexing:** FAISS stores vector representations for semantic retrieval.
6. **Hybrid Retrieval:** FAISS and BM25 retrieve relevant document passages.
7. **Answer Generation:** Groq generates responses using the retrieved context.
8. **Chat Interface:** Users interact with the system through a web-based interface.

## Project Structure

```text
Rag-powered-document-chatbot/
│
├── Backend/
│   ├── app.py
│   ├── chunking.py
│   ├── docling_parser.py
│   ├── document_parser.py
│   ├── rag_service.py
│   └── retriever.py
│
├── Frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── .env.example
├── .gitignore
├── requirements.txt
├── build_faiss.py
├── chunk_documents.py
├── rag_chat.py
├── retrieve_test.py
└── test files
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/imtiaz1425/Rag-powered-document-chatbot.git
cd Rag-powered-document-chatbot
```

### 2. Create a virtual environment

**Windows:**

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root using `.env.example` as a reference.

Add your Groq API key according to the variable names in `.env.example`.

Get an API key from: https://console.groq.com/keys

**Important:** Never upload your `.env` file or expose your API key publicly.

### 5. Run the application

From the project root:

```bash
python -m Backend.app
```

Open your browser:

```text
http://127.0.0.1:5000
```

## Testing

The repository includes test scripts for document processing, retrieval reliability, and the RAG pipeline.

Run the relevant tests using Python, for example:

```bash
python -m pytest
```

## Security

- API credentials are stored in environment variables.
- `.env` is excluded using `.gitignore`.
- Private documents and generated vector indexes are excluded from version control.

## Author

**Imthiyaz Ahmed**

Computer Science and Engineering  
United International University

## License

This project is currently intended for academic and personal use.

