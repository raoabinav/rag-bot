import os
from dotenv import load_dotenv


load_dotenv()

# Extract environment variables safely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1-aws")


REQUIRED_VARS = {
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "PINECONE_API_KEY": PINECONE_API_KEY,
    "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME,
}

for var_name, value in REQUIRED_VARS.items():
    if value is None:
        raise EnvironmentError(f"Missing required env variable: {var_name}")
