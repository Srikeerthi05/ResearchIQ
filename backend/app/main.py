
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from backend.app.rag import answer_question
from backend.app.paper_search import build_search_index


# Create FastAPI application
app = FastAPI(
    title="ResearchIQ API",
    description="AI-powered Multi-Document Research Assistant",
    version="2.0.0"
)


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


# Request structure for asking questions
class QuestionRequest(BaseModel):

    document_id: str = Field(
        min_length=1,
        description="ID of the uploaded research paper"
    )

    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Question about the research paper"
    )


# Root endpoint
@app.get("/")
def home():

    return {
        "message": "Welcome to ResearchIQ API!",
        "status": "Running"
    }


# Upload research paper
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):

    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Read uploaded file
    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded PDF is empty."
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 20 MB upload limit."
        )

    # Check PDF signature
    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file."
        )

    # Generate unique document ID
    document_id = uuid4().hex

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    pdf_path = UPLOAD_DIR / f"{document_id}.pdf"

    try:

        # Save uploaded PDF
        pdf_path.write_bytes(content)

        # Build and save its FAISS index
        index, chunks = build_search_index(
            pdf_path,
            document_id
        )

        return {
            "message": "PDF uploaded and indexed successfully!",
            "document_id": document_id,
            "filename": file.filename,
            "total_chunks": len(chunks),
            "status": "Ready for questions"
        }

    except ValueError as e:

        pdf_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except Exception as e:

        pdf_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )


# Ask questions about a selected paper
@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        result = answer_question(
            request.question,
            request.document_id
        )

        return {
            "document_id": request.document_id,
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except FileNotFoundError:

        raise HTTPException(
            status_code=404,
            detail="Document index not found. Please upload the PDF first."
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"ResearchIQ processing failed: {str(e)}"
        )
