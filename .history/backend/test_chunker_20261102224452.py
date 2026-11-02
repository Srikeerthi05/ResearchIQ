
from pathlib import Path
from app.pdf_processor import extract_text_from_pdf
from app.chunker import chunk_text

project_root = Path(__file__).resolve().parents[1]

pdf_path = (
    project_root
    / "data"
    / "papers"
    / "AttentionIsYourAll.pdf"
)

pages = extract_text_from_pdf(str(pdf_path))

all_chunks = []

for page in pages:
    chunks = chunk_text(page["text"])

    for chunk in chunks:
        all_chunks.append({
            "page_number": page["page_number"],
            "text": chunk
        })

print("Total pages:", len(pages))
print("Total chunks:", len(all_chunks))

print("\nFirst chunk:")
print(all_chunks[0])
print("\nPage number:", all_chunks[0]["page_number"])
