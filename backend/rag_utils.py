# rag_utils.py


from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pinecone import Pinecone, Vector
import os

# Load from environment variables 
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is not set. Check your .env file.")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")

# Initialize OpenAI and Pinecone clients
openai_client = OpenAI(api_key=OPENAI_KEY)
pinecone_client = Pinecone(api_key=PINECONE_KEY)
index = pinecone_client.Index(PINECONE_INDEX_NAME)

# 1. Convert chunks to embeddings
def get_embeddings(chunks: list[str]) -> list[list[float]]:
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=chunks
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print("Error in get_embeddings:", e)
        return []

# 2. Upsert to Pinecone
def upsert_chunks(chunks: list[str], embeddings: list[list[float]], namespace: str):
    try:
        vectors = [
            Vector(
                id=f"doc1:{i}",
                values=emb,
                metadata={"text": chunk}
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        index.upsert(vectors=vectors, namespace=namespace)
    except Exception as e:
        print("Error in upsert_chunks:", e)

# 3. Query Pinecone
def retrieve_relevant_chunks(query: str, namespace: str, top_k: int = 5) -> list[str]:
    
    embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    ).data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    matches = getattr(results, "matches", [])

    return [match.metadata["text"] for match in matches]

# 4. Ask ChatGPT with context
def ask_with_context(user_message: str, context_chunks: list[str]) -> str:
    try:
        context_block = "\n---\n".join(context_chunks)
        messages: list[ChatCompletionMessageParam] = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
]
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content or "No response from model."
    except Exception as e:
        print("Error in ask_with_context:", e)
        return "Something went wrong."
    

