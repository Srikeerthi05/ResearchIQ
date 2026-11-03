
from app.paper_search import load_or_build_search_index, search_paper
from app.llm import generate_answer


def answer_question(question):

    # Step 1: Build the FAISS search index
    index, all_chunks = load_or_build_search_index()

    # Step 2: Retrieve the top 3 relevant chunks
    results = search_paper(
        question,
        index,
        all_chunks,
        k=3
    )

    # Step 3: Handle empty search results
    # Minimum similarity required to answer
    SIMILARITY_THRESHOLD = 0.30

    if not results or results[0]["similarity"] < SIMILARITY_THRESHOLD:
        return {
            "answer": "I couldn't find relevant information in the research paper to answer this question.",
            "sources": []
        }

    # Step 4: Prepare context with page references
    context = ""

    for result in results:
        context += (
            f"\n[Page {result['page_number']}]\n"
            f"{result['text']}\n"
        )

    # Step 5: Create the prompt for the LLM
    prompt = f"""
You are ResearchIQ, an AI research assistant.

Your task is to answer questions using ONLY the research paper
context provided below.

Rules:
1. Do not use outside knowledge.
2. If the answer is not available in the context,
   clearly say that the paper does not provide the answer.
3. Include page references using [Page N].
4. Never invent facts or page numbers.

Research Paper Context:
{context}

User Question:
{question}

Answer:
"""

    # Step 6: Generate the answer using Qwen
    answer = generate_answer(prompt)

    # Step 7: Collect source information independently
    sources = []

    for result in results:
        sources.append({
            "page": result["page_number"],
            "similarity": round(result["similarity"], 4)
        })

    # Step 8: Return answer and sources
    return {
        "answer": answer,
        "sources": sources
    }


# Run ResearchIQ from the terminal
if __name__ == "__main__":

    question = input("Ask ResearchIQ: ")

    result = answer_question(question)

    print("\nResearchIQ Answer:\n")
    print(result["answer"])

    print("\nSources Retrieved:\n")

    if result["sources"]:
        for source in result["sources"]:
            print(
                f"Page: {source['page']} | "
                f"Similarity: {source['similarity']}"
            )
    else:
        print("No sources found.")
