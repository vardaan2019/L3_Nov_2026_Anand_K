from fastapi import FastAPI, APIRouter

from app.aws_rag_pipeline import get_aws_answer

app = FastAPI(title="RAG App")
router = APIRouter()

@app.get("/answer/")
def get_answer(question: str):
    response = get_aws_answer(question)
    print("Response from RAG pipeline:", response)
    return response

# Run below command to start the server
# py -m uvicorn app.aws_main:app --reload --port 8000