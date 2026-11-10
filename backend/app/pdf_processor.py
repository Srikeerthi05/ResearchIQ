
import fitz
from pathlib import Path


def extract_text_from_pdf(pdf_path: str):

    pdf_path = Path(pdf_path)

    # Validate file existence
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}"
        )

    # Validate file extension
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported"
        )

    pages = []

    # Safely open and close PDF
    with fitz.open(pdf_path) as document:

        if len(document) == 0:
            raise ValueError(
                "The PDF contains no pages."
            )

        for page_number, page in enumerate(
            document,
            start=1
        ):

            text = page.get_text("text").strip()

            # Skip pages without extractable text
            if not text:
                continue

            pages.append({
                "page_number": page_number,
                "text": text
            })

    # Detect PDFs with no extractable text
    if not pages:
        raise ValueError(
            "No extractable text found. "
            "The PDF may contain scanned images "
            "and require OCR."
        )

    return pages
