import shutil
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel

import query_hybrid
from ingest_hybrid import ingest_graph, ingest_vector

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
CURRENT_DOC_MARKER = UPLOAD_DIR / "current_document.txt"

load_dotenv(BASE_DIR / ".env")

app = FastAPI(title="Spotify Architecture RAG API")

_cached_app = None


def get_cached_app():
    global _cached_app
    if _cached_app is None:
        _cached_app = query_hybrid.build_self_rag_app()
    return _cached_app


def invalidate_cache() -> None:
    global _cached_app
    _cached_app = None


def has_ingested_data() -> bool:
    return any(query_hybrid.INDEX_DIR.glob("page_*"))


def get_current_document() -> Optional[str]:
    if CURRENT_DOC_MARKER.exists():
        return CURRENT_DOC_MARKER.read_text().strip()
    return None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    vector_context: str
    graph_context: str
    retrieval_attempts: int
    generation_attempts: int
    generation_grade: str
    out_of_scope: bool


class StatusResponse(BaseModel):
    has_document: bool
    current_document: Optional[str] = None


class IngestResponse(BaseModel):
    message: str
    filename: str
    pages: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    return StatusResponse(has_document=has_ingested_data(), current_document=get_current_document())


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    if not has_ingested_data():
        raise HTTPException(status_code=400, detail="No document has been ingested yet. Upload one via /ingest first.")

    try:
        app_graph = get_cached_app()
        result = query_hybrid.run_self_rag(request.question, app=app_graph)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        answer=result["answer"],
        vector_context=result["vector_context"],
        graph_context=result["graph_context"],
        retrieval_attempts=result["retrieval_attempts"],
        generation_attempts=result["generation_attempts"],
        generation_grade=result["generation_grade"],
        out_of_scope=result["out_of_scope"],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        for old_pdf in UPLOAD_DIR.glob("*.pdf"):
            old_pdf.unlink()

        dest_path = UPLOAD_DIR / file.filename
        dest_path.write_bytes(file.file.read())

        if query_hybrid.INDEX_DIR.exists():
            shutil.rmtree(query_hybrid.INDEX_DIR)

        pages = PyPDFLoader(str(dest_path)).load()

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        ingest_vector(pages, embeddings)
        ingest_graph(pages, llm)

        CURRENT_DOC_MARKER.write_text(file.filename)
        invalidate_cache()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(message="Document processed", filename=file.filename, pages=len(pages))
