
from pathlib import Path
from app.pdf_processor import extract_text_from_pdf

# Find the main ResearchIQ project directory
project_root = Path(__file__).resolve().parents[1]

# Build the PDF path
pdf_path = project_root / "data" / "papers" / "attention_is_all_you_need.pdf"

pages = extract_text_from_pdf(str(pdf_path))

print("Total pages:", len(pages))

for page in pages[:2]:

    print("\nPAGE:", page["page_number"])

    print(page["text"][:500])
