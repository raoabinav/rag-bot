"""
Chat service for handling chat-related business logic.
"""
from typing import List, Dict, Any
from ..utils.openai_utils import openai_utils
from ..utils.pinecone_utils import pinecone_utils

class ChatService:
    def __init__(self, openai_utils=openai_utils, pinecone_utils=pinecone_utils):
        """Initialize chat service with utility instances."""
        self.openai_utils = openai_utils
        self.pinecone_utils = pinecone_utils
    
    def get_relevant_context(self, query: str, namespace: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a query from Pinecone.
        
        Args:
            query: The user's query
            namespace: Pinecone namespace to search in
            top_k: Number of results to return
            
        Returns:
            List of relevant context chunks with metadata
        """
        print(f"Getting relevant context for query: {query}")
        
        # Get query embedding
        query_embedding = self.openai_utils.get_embeddings([query])
        if not query_embedding:
            print("Failed to generate query embedding")
            return []
            
        # Query Pinecone
        results = self.pinecone_utils.query_index(
            query_embedding=query_embedding[0],
            namespace=namespace,
            top_k=top_k
        )
        
        return results
    
    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response using the query and retrieved context.
        
        Args:
            query: The user's query
            context_chunks: List of context chunks with metadata
            
        Returns:
            Generated response text
        """
        if not context_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Extract text from context chunks
        context_texts = [chunk.get('metadata', {}).get('text', '') for chunk in context_chunks]
        context_block = "\n---\n".join(context_texts)
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
        If the answer cannot be found in the context, say "I don't know" instead of making up an answer."""
        
        user_prompt = f"""Context information is below.
        --------------------
        {context_block}
        --------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {query}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.openai_utils.get_chat_completion(messages)
    
    def process_chat_message(self, message: str, namespace: str) -> str:
        """
        Process a chat message end-to-end.
        
        Args:
            message: The user's message
            namespace: Pinecone namespace to use
            
        Returns:
            Generated response
        """
        # Get relevant context
        context_chunks = self.get_relevant_context(message, namespace)
        
        # Generate response
        return self.generate_response(message, context_chunks)

# Create a default instance for convenience
chat_service = ChatService()
