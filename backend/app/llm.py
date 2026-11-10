
import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"


def generate_answer(prompt):

    payload = {
        "model": "qwen2.5:1.5b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42
        }
    }

    response = httpx.post(
        OLLAMA_URL,
        json=payload,
        timeout=120.0
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]


if __name__ == "__main__":

    question = "Explain the Transformer architecture in simple terms."

    answer = generate_answer(question)

    print("\nQwen's Answer:\n")
    print(answer)
