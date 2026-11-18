from app.aws_vector_store import retrieve_from_kb, generate_answer

def get_aws_answer(user_query: str):
# Step 1: Retrieve documents from KB
    kb_results = retrieve_from_kb(user_query)
    if not kb_results:
        print("No relevant documents found in Knowledge Base.")

    # Combine retrieved documents into context
    context_text = "\n\n".join(
        [doc["content"]["text"] for doc in kb_results if "content" in doc and "text" in doc["content"]]
    )

    # Step 2: Generate answer using retrieved context
    answer = generate_answer(context_text, user_query)

    print("\n--- Retrieved Context ---")
    print(context_text)
    print("\n--- Generated Answer ---")
    print(answer)
    return answer