# STEP 3: Create vector embeddings using FAISS

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from app.embeddings import get_chunks
from app.env import get_env_variable
import os
import boto3
import json

aws_access_key = get_env_variable('AWS_ACCESS_KEY_ID')
aws_secret_access_key = get_env_variable('AWS_SECRET_ACCESS_KEY')

# ---------- CONFIGURATION ----------
AWS_REGION = "us-east-1"  # Change to your Bedrock region
KNOWLEDGE_BASE_ID = "R2I3X3ULDD"  # Replace with your Knowledge Base ID
MODEL_ID = "amazon.titan-embed-text-v2:0"  # Example Bedrock model
MAX_TOKENS = 512
TEMPERATURE = 0.5

# ---------- INITIALIZE CLIENTS ----------
try:
    bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION,
                                        aws_access_key_id=aws_access_key,
                                            aws_secret_access_key=aws_secret_access_key)
    bedrock_runtime_client = boto3.client("bedrock-runtime", region_name=AWS_REGION,
                                          aws_access_key_id=aws_access_key,
                                              aws_secret_access_key=aws_secret_access_key)
except Exception as e:
    print(f"Error initializing AWS clients: {e}")
    
# ---------- FUNCTION: Retrieve from Knowledge Base ----------
def retrieve_from_kb(query_text):
    """
    Retrieves relevant documents from AWS Bedrock Knowledge Base.
    """
    try:
        response = bedrock_agent_client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query_text}
        )
        return response.get("retrievalResults", [])
    except Exception as e:
        print(f"Error retrieving from Knowledge Base: {e}")
        return []

# ---------- FUNCTION: Generate Answer ----------
def generate_answer(context_text, user_query):
    """
    Sends context + user query to Bedrock model for generation.
    """

    prompt = (
        f"Use the following pieces of the context to answer user's question.\n"
        f"If you don't know the answer, just say you don't know, don't try to make up the answer.\n"
        f"------------------------\n"
        f"Context: {context_text}\n\n"
        f"Question: {user_query}\n\n"
        f"Answer: "
    )


    try:
        body = {
        "messages": [
        {
            "role": "user",
            "content": [
                {
                    "text": prompt
                }
            ]
        }
        ],
        "inferenceConfig": {
        "maxTokens": 1000,
        "temperature": 0.7,
        "topP": 0.9
        }
        }

        print("\n--- Prompt Sent to Bedrock Model ---", body)
        print("\n--- Body JSON ---", json.dumps(body))


        response = bedrock_runtime_client.invoke_model(
            modelId="global.amazon.nova-2-lite-v1:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        #print("\n--- Raw Response from Bedrock Model ---", response)
        result = json.loads(response["body"].read())
        #print(result.keys())
        #print("\n", result["output"])
        # Extract generated text
        generated_text = result["output"]["message"]["content"][0]["text"]

        return generated_text
    
    except Exception as e:
        print(f"Error generating answer: {e}")
        return ""