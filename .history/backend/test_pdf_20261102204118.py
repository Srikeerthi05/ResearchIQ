from app.pdf_processor import extract_text_from_pdf


pdf_path = "data/papers/attention_is_all_you_need.pdf"

pages = extract_text_from_pdf(pdf_path)

print("Total pages:", len(pages))

for page in pages[:2]:

    print("\nPAGE:", page["page_number"])

    print(page["text"][:500])