from fastapi import FastAPI, APIRouter

from app.rag_pipeline import get_rag_pipeline

app = FastAPI(title="RAG App")
retriever, qa_chain = get_rag_pipeline()
router = APIRouter()

@app.get("/answer/")
def get_answer(question: str):
    response = qa_chain({"query": question})
    return response["result"]

# Run below command to start the server
# py -m uvicorn app.main:app --reload --port 8000
# Browser URL: http://localhost:8000/answer/?question=What%20is%20an%20artificial%20intelligence
