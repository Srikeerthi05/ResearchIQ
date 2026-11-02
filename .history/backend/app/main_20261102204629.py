
from fastapi import FastAPI

app = FastAPI(
    title="ResearchIQ API",
    description="AI-powered research intelligence platform",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Welcome to ResearchIQ!",
        "status": "running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
