# ResearchIQ
### AI-Powered Research Paper Assistant Using Retrieval-Augmented Generation (RAG)

ResearchIQ is an AI-powered research paper assistant that enables users to ask natural-language questions about research documents and receive context-aware answers grounded in the content of the paper.

The application combines semantic search, vector similarity retrieval, and Large Language Model (LLM) generation to make academic research easier to explore and understand.

Instead of relying entirely on an LLM's pre-trained knowledge, ResearchIQ retrieves relevant passages from a research paper and supplies them as context to the language model before generating a response. This approach helps produce more relevant, document-grounded answers and provides references to the retrieved source pages.

---

## 1. Project Overview

Research papers often contain complex terminology, lengthy explanations, mathematical concepts, and extensive technical discussions. Finding specific information manually can be time-consuming.

ResearchIQ addresses this problem through a Retrieval-Augmented Generation pipeline.

Users can ask questions such as:

- What is the role of self-attention in the Transformer architecture?
- How does multi-head attention work?
- What problem does the Transformer architecture address?
- What are the key contributions of the paper?

The system retrieves relevant text segments from the research paper, evaluates their semantic similarity to the query, and uses the retrieved information to generate an answer.

### Key Features

- **PDF Text Extraction:** Extracts textual content and page information from research papers using PyMuPDF.
- **Intelligent Text Chunking:** Divides extracted content into smaller, overlapping text segments to improve retrieval.
- **Semantic Embeddings:** Converts text segments and user queries into numerical vector representations using Sentence Transformers.
- **Vector Similarity Search:** Uses FAISS to identify relevant passages based on semantic similarity.
- **Retrieval-Augmented Generation:** Combines retrieved document context with an LLM to generate answers.
- **Local LLM Integration:** Uses Ollama to run the Qwen 2.5 1.5B language model locally.
- **Source Retrieval:** Maintains page-level metadata and returns retrieved source pages with similarity scores.
- **Similarity Thresholding:** Filters low-relevance retrieval results to help handle unrelated questions.
- **Persistent Vector Storage:** Saves the generated FAISS index and associated chunk metadata to avoid rebuilding the index unnecessarily.
- **REST API:** Exposes the question-answering pipeline through FastAPI.
- **API Documentation:** Provides interactive API testing through automatically generated Swagger UI documentation.

---

## 2. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Programming Language | Python | Core application development |
| PDF Processing | PyMuPDF | Extracting text from PDF documents |
| Text Processing | Python | Chunking and preprocessing |
| Embedding Model | Sentence Transformers | Generating semantic vector embeddings |
| Embedding Model Name | all-MiniLM-L6-v2 | Converting text into 384-dimensional embeddings |
| Vector Database / Search | FAISS | Efficient similarity-based vector retrieval |
| Numerical Processing | NumPy | Managing embedding arrays |
| Large Language Model | Qwen 2.5 1.5B | Generating natural-language answers |
| LLM Runtime | Ollama | Running the language model locally |
| HTTP Client | HTTPX | Communicating with the Ollama API |
| Backend Framework | FastAPI | Building REST API endpoints |
| API Server | Uvicorn | Running the FastAPI application |
| Data Validation | Pydantic | Validating incoming API requests |
| Development Environment | VS Code | Application development and debugging |
| Version Control | Git and GitHub | Source code management |

---

## 3. System Architecture

ResearchIQ follows a Retrieval-Augmented Generation architecture consisting of two primary stages: document ingestion and question answering.

### A. Document Ingestion Pipeline

The ingestion pipeline transforms a research paper into searchable vector representations.

1. **PDF Loading:** The application loads the research paper from the configured document location.

2. **Text Extraction:** PyMuPDF extracts text from the PDF while preserving page-level information.

3. **Text Chunking:** Extracted content is divided into smaller text segments. Overlapping sections help preserve contextual continuity between chunks.

4. **Embedding Generation:** Sentence Transformers converts each chunk into a 384-dimensional vector representation using the `all-MiniLM-L6-v2` model.

5. **Vector Index Creation:** FAISS stores the generated embeddings in an `IndexFlatIP` index.

6. **Normalization and Similarity:** Embeddings are normalized before indexing. Inner product similarity between normalized vectors is used as cosine similarity.

7. **Persistent Storage:** The FAISS index and associated chunk metadata are stored locally for reuse.

### B. Question-Answering Pipeline

When a user submits a question, the following process takes place:

1. **Query Input:** The user submits a natural-language question through the API or terminal interface.

2. **Query Embedding:** The question is converted into a vector using the same embedding model used for document chunks.

3. **Semantic Retrieval:** FAISS searches the vector index and retrieves the most relevant text segments.

4. **Relevance Filtering:** Retrieved results are evaluated using similarity scores and a configured relevance threshold.

5. **Context Construction:** Relevant passages are combined with their corresponding page references.

6. **Prompt Engineering:** The retrieved passages and user question are included in a structured prompt that instructs the language model to answer using the provided research paper context.

7. **LLM Generation:** Qwen 2.5 1.5B generates the response through the local Ollama runtime.

8. **Response Delivery:** The application returns the generated answer and retrieved source information.

---

## 4. Architecture Diagram

```text
                  RESEARCHIQ
                       |
             Research Paper PDF
                       |
                       v
                PDF Extraction
                   PyMuPDF
                       |
                       v
                 Text Chunking
                       |
                       v
              Sentence Transformers
                 Embeddings
                       |
                       v
                  FAISS Index
                       |
                Persistent Storage
                       |
                       |
              User Submits Question
                       |
                       v
                Query Embedding
                       |
                       v
               Semantic Retrieval
                       |
                       v
                Relevance Filtering
                       |
                       v
                Context Construction
                       |
                       v
                 Prompt Creation
                       |
                       v
                  Ollama Runtime
                       |
                       v
                Qwen 2.5 1.5B
                       |
                       v
             Generated Answer + Sources
```

---

## 5. Project Structure

```text
ResearchIQ/
│
├── backend/
│   │
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── pdf_processor.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── paper_search.py
│   │   ├── llm.py
│   │   └── rag.py
│   │
│   ├── data/
│   │   └── papers/
│   │
│   └── tests/
│
├── data/
│   └── index/
│
├── docs/
├── notebooks/
├── tests/
├── .gitignore
├── requirements.txt
└── README.md
```

*Note: The structure represents the intended organization of the project. Some directories and modules may be added or reorganized as development progresses.*

---

## 6. Core Components

### PDF Processor

Responsible for extracting text from PDF documents.

It uses PyMuPDF to extract content page by page, allowing the application to associate each extracted text segment with its original page number.

### Text Chunker

Splits extracted text into smaller segments using a configurable chunk size and overlap.

Chunking improves retrieval by allowing the system to identify specific sections of a document instead of processing an entire paper as one large text block.

### Embedding Model

ResearchIQ uses `all-MiniLM-L6-v2` from Sentence Transformers.

The model converts textual content into 384-dimensional dense vector representations.

Since both document chunks and user queries are embedded using the same model, they can be compared in a shared semantic vector space.

### FAISS Vector Search

FAISS is used for efficient similarity-based retrieval.

The project uses `IndexFlatIP`, which performs exact inner-product similarity search.

Since embeddings are normalized, inner-product scores correspond to cosine similarity.

### RAG Pipeline

The RAG module coordinates retrieval and generation.

It retrieves relevant passages, constructs a context-aware prompt, sends the prompt to the language model, and returns the generated answer along with source information.

### LLM Integration

ResearchIQ integrates Qwen 2.5 1.5B through Ollama.

The model runs locally and receives prompts containing retrieved research paper passages.

This avoids requiring a paid external LLM API for the current implementation.

### FastAPI Backend

FastAPI exposes the application functionality through HTTP endpoints.

The API layer is separated from the underlying RAG implementation, allowing the AI pipeline to be accessed independently of the eventual frontend.

---

## 7. API Documentation

The backend currently exposes the following endpoints.

### GET /

Checks whether the ResearchIQ API is running.

**Endpoint:**

```http
GET /
```

**Example response:**

```json
{
    "message": "Welcome to ResearchIQ API!",
    "status": "Running"
}
```

### POST /ask

Accepts a natural-language question and returns an answer generated using the RAG pipeline.

**Endpoint:**

```http
POST /ask
```

**Request Body:**

```json
{
    "question": "What is the role of self-attention in the Transformer architecture?"
}
```

**Example response structure:**

```json
{
    "question": "What is the role of self-attention in the Transformer architecture?",
    "answer": "Generated answer based on the retrieved research paper context.",
    "sources": [
        {
            "page": 5,
            "similarity": 0.6527
        }
    ]
}
```

The actual response content and source scores depend on the query and retrieved passages.

### Interactive API Documentation

FastAPI automatically generates interactive API documentation using Swagger UI.

After starting the backend, access:

```text
http://127.0.0.1:8000/docs
```

Use the documentation interface to inspect endpoints and submit test requests.

---

## 8. Installation and Setup

### Prerequisites

Install the following before running the application:

- Python 3.10 or later
- Git
- Ollama
- Visual Studio Code (recommended)

### Step 1: Clone the Repository

```bash
git clone (https://github.com/Srikeerthi05/ResearchIQ)
cd ResearchIQ
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Install and Configure Ollama

Install Ollama from:

https://ollama.com/

Download the required model:

```bash
ollama pull qwen2.5:1.5b
```

Verify that the model is available:

```bash
ollama list
```

### Step 5: Add a Research Paper

Place the research paper PDF in the configured document directory.

The current development configuration uses:

```text
data/papers/AttentionIsAllYouNeed.pdf
```

Make sure the filename and path match the configuration in the PDF ingestion or search module.

### Step 6: Run the Application

Navigate to the backend directory:

```bash
cd backend
```

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

The application should be available at:

```text
http://127.0.0.1:8000
```

### Step 7: Test the API

Open the interactive documentation:

```text
http://127.0.0.1:8000/docs
```

Expand `POST /ask`, select **Try it out**, provide a question, and execute the request.

---

## 9. Retrieval and Relevance

ResearchIQ uses semantic similarity rather than relying exclusively on keyword matching.

For example, a question about the purpose of attention mechanisms can retrieve relevant passages even when the document uses different wording.

The retrieval process uses normalized embeddings and cosine similarity through FAISS inner-product search.

A configurable similarity threshold helps filter unrelated queries.

However, similarity scores are not guaranteed measures of factual correctness. Thresholds must be evaluated against representative relevant and irrelevant questions.

---

## 10. Hallucination Control and Source Grounding

An important design goal of ResearchIQ is to reduce unsupported responses.

The current implementation uses several grounding mechanisms:

- Retrieved document passages are included in the LLM prompt.
- The prompt instructs the model to answer using the supplied context.
- The model is instructed to acknowledge when the paper does not contain the requested information.
- Similarity thresholding filters low-relevance retrieval results.
- Page metadata is maintained alongside retrieved chunks.

These mechanisms improve grounding but do not guarantee that every generated statement is correct.

Further improvements will include automated citation verification, answer-to-source validation, and systematic hallucination evaluation.

---

## 11. Current Implementation Status

The following functionality has been implemented and tested:

- PDF text extraction.
- Text chunking with overlap.
- Semantic embedding generation.
- FAISS-based similarity retrieval.
- Local LLM integration through Ollama.
- End-to-end RAG question answering.
- Page-level source metadata retrieval.
- Similarity-based relevance filtering.
- Persistent vector index and chunk storage.
- FastAPI application setup.
- GET `/` endpoint.
- POST `/ask` endpoint.
- Interactive Swagger API testing.

---

## 12. Future Enhancements

ResearchIQ is being developed further with the following planned improvements:

**Document Management**
- Support uploading research papers through the API.
- Support multiple PDF documents.
- Maintain independent indexes or document metadata for different papers.
- Implement document deletion and re-indexing.

**Advanced Retrieval**
- Experiment with alternative chunking strategies.
- Implement hybrid search combining semantic and keyword retrieval.
- Explore reranking models for improved retrieval precision.
- Optimize retrieval parameters using evaluation datasets.

**Answer Quality**
- Implement citation verification.
- Improve context selection and prompt construction.
- Evaluate hallucination rates.
- Add structured answer evaluation and retrieval metrics.

**Application Development**
- Develop a React-based frontend.
- Build a conversational interface.
- Add document management and source navigation.
- Improve error handling and request validation.

**Deployment**
- Containerize the application using Docker.
- Deploy the backend.
- Add logging and monitoring.
- Improve performance and scalability.

---

## 13. Learning Outcomes

This project demonstrates practical implementation of several important AI engineering concepts:

- Retrieval-Augmented Generation architecture.
- Semantic search and dense vector representations.
- Text preprocessing and chunking strategies.
- Vector indexing and similarity search using FAISS.
- Integration of locally hosted Large Language Models.
- Prompt engineering and context grounding.
- Building REST APIs with FastAPI.
- Request validation using Pydantic.
- Separating AI processing logic from API and application layers.
- Persistent storage of vector indexes and metadata.

---

## 14. Acknowledgements and References

- [Sentence Transformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Ollama](https://ollama.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Qwen Models](https://huggingface.co/Qwen)

---

## Project Vision

ResearchIQ aims to make technical research more accessible by combining semantic retrieval with grounded language generation.

The long-term goal is to develop a research assistant capable of processing multiple academic documents, retrieving precise evidence, and providing reliable, source-supported explanations through an intuitive interface.
