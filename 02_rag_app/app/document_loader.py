# STEP 1: Load the external knowledgebase and convert into pages.
# Example - Knowledgebase in pdf format
from langchain_community.document_loaders import PyPDFLoader
from app.env import get_env_variable

def get_knowledgebase():
    loader = PyPDFLoader("Artificial-Intelligence.pdf")
    pdf_pages = loader.load()
    print(f'No of pages: {len(pdf_pages)}')
    return pdf_pages