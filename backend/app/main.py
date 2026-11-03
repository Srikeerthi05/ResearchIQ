
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag import answer_question


# Create FastAPI application
app = FastAPI(
    title="ResearchIQ API",
    description="AI-powered Research Paper Assistant",
    version="1.0.0"
)


# Define the incoming request structure
class QuestionRequest(BaseModel):

    question: str = Field(
        min_length=3,
        max_length=1000,
        description="Question about the uploaded research paper"
    )


# Root endpoint
@app.get("/")
def home():

    return {
        "message": "Welcome to ResearchIQ API!",
        "status": "Running"
    }


# Question-answering endpoint
@app.post("/ask")
def ask_question(request: QuestionRequest):

    try:

        result = answer_question(request.question)

        return {
            "question": request.question,
            "answer": result["answer"],
            "sources": result["sources"]
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"ResearchIQ processing failed: {str(e)}"
        )
