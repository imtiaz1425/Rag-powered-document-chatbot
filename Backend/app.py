import os
import sys
import shutil
import traceback
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from Backend.rag_service import DocumentChatService

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB max file size

UPLOAD_DIR = BASE_DIR / "data" / "uploads"
FRONTEND_DIR = BASE_DIR / "Frontend"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".docx"}

# Initialize the document chat service singleton
chat_service = DocumentChatService()


@app.after_request
def add_cors_headers(response):
    """Enable CORS for decoupled deployments (e.g. Frontend on Netlify, Backend on Render)."""
    allowed_origin = os.getenv("CORS_ORIGIN", "*")
    response.headers["Access-Control-Allow-Origin"] = allowed_origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.before_request
def handle_preflight():
    """Respond immediately to HTTP OPTIONS preflight requests."""
    if request.method == "OPTIONS":
        return "", 204


def is_allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/style.css", methods=["GET"])
def style_css():
    return send_from_directory(str(FRONTEND_DIR), "style.css")


@app.route("/script.js", methods=["GET"])
def script_js():
    return send_from_directory(str(FRONTEND_DIR), "script.js")


@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint for Render/cloud deployment monitoring."""
    return jsonify({"status": "healthy", "service": "AI Document Assistant"}), 200


@app.route("/status", methods=["GET"])
@app.route("/api/status", methods=["GET"])
def status():
    """Return the current document session status."""
    return jsonify(chat_service.session_info)


@app.route("/upload", methods=["POST"])
@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    Handle document upload, extraction, chunking, and indexing.
    Replaces the previous active document session safely.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part in request."}), 400

        f = request.files["file"]
        if not f or not f.filename:
            return jsonify({"error": "No file selected."}), 400

        if not is_allowed_file(f.filename):
            ext = Path(f.filename).suffix
            return jsonify({
                "error": f"Unsupported format '{ext}'. Please upload a PDF (.pdf) or Word (.docx) document."
            }), 400

        # Save to uploads directory
        save_path = UPLOAD_DIR / f.filename
        f.save(str(save_path))

        # Process and index the document through our RAG pipeline
        result = chat_service.process_and_index_document(save_path)
        return jsonify({
            "status": "success",
            "filename": result["filename"],
            "file_type": result["file_type"],
            "pages": result["pages"],
            "chunks": result["chunks"],
            "parser_used": result.get("parser_used", "docling"),
            "message": result["message"]
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
@app.route("/api/ask", methods=["POST"])
def ask_question():
    """
    Handle user questions against the active document with grounded RAG and citations.
    """
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question is required."}), 400

        response = chat_service.answer_question(question)
        if "error" in response and not response.get("answer"):
            return jsonify(response), 400

        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
