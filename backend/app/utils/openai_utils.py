import os
from typing import List, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

class OpenAIUtils:
    def __init__(self, api_key: Optional[str] = None):
        """Initialize OpenAI client with optional API key."""
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def get_embeddings(self, texts: List[str], model: str = "text-embedding-3-small") -> List[List[float]]:
        """
        Get embeddings for a list of text strings.
        
        Args:
            texts: List of text strings to embed
            model: Name of the embedding model to use
            
        Returns:
            List of embedding vectors
        """
        try:
            clean_texts = [text for text in texts if text.strip()]
            if not clean_texts:
                print("No valid text provided for embedding")
                return []
                
            response = self.client.embeddings.create(
                model=model,
                input=clean_texts
            )
            
            print(f"Generated {len(response.data)} embeddings for {len(clean_texts)} texts")
            return [item.embedding for item in response.data]
            
        except Exception as e:
            print(f"Error in get_embeddings: {e}")
            return []
    
    def get_chat_completion(
        self, 
        messages: List[ChatCompletionMessageParam], 
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """
        Get chat completion from OpenAI's API.
        
        Args:
            messages: List of message dictionaries
            model: Name of the model to use
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content or ""
            
        except Exception as e:
            print(f"Error in get_chat_completion: {e}")
            return ""

# Create a default instance for convenience
openai_utils = OpenAIUtils()
