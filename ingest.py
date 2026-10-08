from rag import load_and_split_documents, create_vector_store

file_path = "documents/company_policies.txt"

chunks = load_and_split_documents(file_path)
print(f"Loaded chunks: {len(chunks)}")

vectorstore = create_vector_store(chunks)
print("Ingest complete. Vector store updated.")