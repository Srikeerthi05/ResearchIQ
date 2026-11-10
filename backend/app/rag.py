
from backend.app.paper_search import (
    load_document_index,
    search_paper
)

from backend.app.llm import generate_answer


def answer_question(question, document_id):

    # Step 1: Load document index
    index, all_chunks = load_document_index(document_id)

    # Step 2: Retrieve relevant chunks
    results = search_paper(
        question,
        index,
        all_chunks,
        k=3
    )

    # Step 3: Check relevance
    SIMILARITY_THRESHOLD = 0.30

    if not results or results[0]["similarity"] < SIMILARITY_THRESHOLD:

        return {
            "answer": (
                "I couldn't find relevant information "
                "in the research paper to answer this question."
            ),
            "sources": [],
            "context": "",
            "retrieved_chunks": []
        }

    # Step 4: Prepare context
    context_parts = []

    for result in results:

        context_parts.append(
            f"\n[Page {result['page_number']}]\n"
            f"{result['text']}\n"
        )

    context = "\n".join(context_parts)

    # Step 5: Build prompt
    prompt = f"""
You are ResearchIQ, an AI research assistant.

Answer using ONLY the research paper context below.

Rules:
1. Do not use outside knowledge.
2. If the answer is unavailable, say so.
3. Include page references using [Page N].
4. Never invent facts or page numbers.
5. Keep the answer clear and concise.

Research Paper Context:
{context}

User Question:
{question}

Answer:
"""

    # Step 6: Generate answer
    answer = generate_answer(prompt)

    # Step 7: Collect sources
    sources = []

    for result in results:

        sources.append({
            "page": result["page_number"],
            "similarity": round(result["similarity"], 4)
        })

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
        "retrieved_chunks": results
    }
