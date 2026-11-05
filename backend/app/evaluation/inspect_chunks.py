from backend.app.paper_search import load_document_index

doc_id = "38eef68c88ea4b2d881d3b3969f27aa4"

index, chunks = load_document_index(doc_id)

target_ids = [6, 7, 10, 11]

for chunk_id in target_ids:
    chunk = chunks[chunk_id]

    print("\nCHUNK ID:", chunk_id)
    print("PAGE:", chunk["page_number"])
    print("TEXT:", chunk["text"])
    print("-" * 70)