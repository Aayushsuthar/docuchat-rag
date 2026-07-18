"""Document ingestion utilities: load raw files and split them into overlapping text chunks."""

from pathlib import Path
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}

def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_pdf_file(file_path):
    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)

def read_any_file(file_path):
    if file_path.suffix.lower() == ".pdf":
        return read_pdf_file(file_path)
    return read_text_file(file_path)

def load_documents(folder_path):
    paths = [p for p in Path(folder_path).rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    documents = [{"source": str(p), "text": read_any_file(p)} for p in paths]
    return documents

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks

def chunk_documents(documents, chunk_size=500, overlap=50):
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size, overlap)
        entries = [{"source": doc["source"], "chunk_id": i, "text": p} for i, p in enumerate(pieces)]
        all_chunks.extend(entries)
    return all_chunks

