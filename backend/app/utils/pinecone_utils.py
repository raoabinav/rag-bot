"""
Pinecone utility functions for vector operations.
"""
import os
from typing import List, Optional, Dict, Any
from pinecone import Pinecone, Vector

class PineconeUtils:
    def __init__(self, api_key: Optional[str] = None, index_name: Optional[str] = None):
        """
        Initialize Pinecone client and index.
        
        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            index_name: Name of the Pinecone index (defaults to PINECONE_INDEX_NAME env var)
        """
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        
        if not self.api_key or not self.index_name:
            raise ValueError("Pinecone API key and index name must be provided or set in environment variables")
            
        self.client = Pinecone(api_key=self.api_key)
        self.index = self.client.Index(self.index_name)
    
    def upsert_vectors(
        self, 
        texts: List[str], 
        embeddings: List[List[float]], 
        namespace: str,
        id_prefix: str = "doc",
        batch_size: int = 50
    ) -> None:
        """
        Upload text chunks and their embeddings to Pinecone.
        
        Args:
            texts: List of text chunks
            embeddings: List of corresponding embedding vectors
            namespace: Pinecone namespace
            id_prefix: Prefix for vector IDs
            batch_size: Number of vectors to upload in each batch
        """
        if not texts or not embeddings:
            print("No texts or embeddings provided for upsert")
            return
            
        if len(texts) != len(embeddings):
            print(f"Mismatch: {len(texts)} texts vs {len(embeddings)} embeddings")
            return
            
        total = len(texts)
        print(f"Starting upload of {total} vectors in batches of {batch_size}...")
        
        try:
            for i in range(0, total, batch_size):
                batch_end = min(i + batch_size, total)
                batch_texts = texts[i:batch_end]
                batch_embeddings = embeddings[i:batch_end]
                
                vectors = [
                    Vector(
                        id=f"{id_prefix}:{i + j}",
                        values=emb,
                        metadata={"text": chunk}
                    )
                    for j, (chunk, emb) in enumerate(zip(batch_texts, batch_embeddings))
                ]
                
                print(f"Uploading batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} "
                      f"(items {i+1}-{batch_end} of {total})...")
                
                try:
                    self.index.upsert(vectors=vectors, namespace=namespace)
                    print(f"✅ Successfully uploaded batch of {len(vectors)} vectors")
                except Exception as batch_error:
                    print(f"❌ Error uploading batch: {batch_error}")
                    raise
                    
        except Exception as e:
            print(f"❌ Error in upsert_vectors: {e}")
            raise
    
    def query_index(
        self,
        query_embedding: List[float],
        namespace: str,
        top_k: int = 5,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query the Pinecone index for similar vectors.
        
        Args:
            query_embedding: The query embedding vector
            namespace: Pinecone namespace to query
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            
        Returns:
            List of query results with metadata
        """
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=include_metadata,
                namespace=namespace
            )
            
            matches = getattr(results, "matches", [])
            print(f"Found {len(matches)} matches")
            
            return [
                {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if hasattr(match, 'metadata') else {}
                }
                for match in matches
            ]
            
        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

# Create a default instance for convenience
pinecone_utils = PineconeUtils()
