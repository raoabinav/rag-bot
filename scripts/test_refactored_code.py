"""
Test script to verify the refactored code works as expected.
"""
import sys
import os
from pathlib import Path

# Add the backend directory to the path
sys.path.append(str(Path(__file__).parent.parent / 'backend'))

# Import the refactored code
from app.utils.openai_utils import OpenAIUtils
from app.utils.pinecone_utils import PineconeUtils
from app.services.chat_service import ChatService

def test_openai_utils():
    """Test OpenAI utilities."""
    print("\n=== Testing OpenAI Utils ===")
    openai = OpenAIUtils()
    
    # Test embeddings
    embeddings = openai.get_embeddings(["test embedding"])
    print(f"Got embeddings: {len(embeddings[0]) if embeddings else 'None'}")
    
    # Test chat completion
    response = openai.get_chat_completion([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello!"}
    ])
    print(f"Chat response: {response[:50]}...")

def test_pinecone_utils():
    """Test Pinecone utilities."""
    print("\n=== Testing Pinecone Utils ===")
    try:
        pinecone = PineconeUtils()
        print("Successfully connected to Pinecone")
        
        # Test query (just check if it doesn't error out)
        results = pinecone.query_index(
            query_embedding=[0.1] * 1536,  # Dummy embedding
            namespace="test-namespace",
            top_k=1
        )
        print(f"Query results: {len(results)} items")
    except Exception as e:
        print(f"Pinecone test skipped or failed: {e}")

def test_chat_service():
    """Test the chat service."""
    print("\n=== Testing Chat Service ===")
    try:
        service = ChatService()
        
        # Test with a simple query
        response = service.process_chat_message(
            message="What is the capital of France?",
            namespace="avengers-bot"
        )
        print(f"Chat response: {response[:100]}...")
    except Exception as e:
        print(f"Chat service test failed: {e}")

if __name__ == "__main__":
    print("Starting tests...")
    
    # Run tests
    test_openai_utils()
    test_pinecone_utils()
    test_chat_service()
    
    print("\nTests completed!")
