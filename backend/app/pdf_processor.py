
import fitz
from pathlib import Path


def extract_text_from_pdf(pdf_path: str):

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")

    document = fitz.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text")

        pages.append({
            "page_number": page_number,
            "text": text.strip()
        })

    document.close()

    return pages
