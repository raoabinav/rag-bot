# rag_utils.py


import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '../secrets.env'))

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pinecone import Pinecone, Vector

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
        # Remove empty strings just in case
        clean_chunks = [c for c in chunks if c.strip()]
        if not clean_chunks:
            print(" All chunks are empty after stripping.")
            return []

        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=clean_chunks
        )
        print(f" Got {len(response.data)} embeddings for {len(clean_chunks)} chunks.")
        embeddings = [item.embedding for item in response.data]

        if len(embeddings) != len(clean_chunks):
            print(f" Mismatch: Got {len(embeddings)} embeddings for {len(clean_chunks)} chunks.")

        return embeddings

    except Exception as e:
        print(" Error in get_embeddings:", e)
        return []


# 2. Upsert to Pinecone with batching
def upsert_chunks(chunks: list[str], embeddings: list[list[float]], namespace: str, id_prefix: str = "doc", batch_size: int = 50):
    """
    Upload chunks to Pinecone in batches with progress tracking.
    
    Args:
        chunks: List of text chunks
        embeddings: List of corresponding embeddings
        namespace: Pinecone namespace
        id_prefix: Prefix for vector IDs
        batch_size: Number of vectors to upload in each batch
    """
    if not chunks or not embeddings:
        print("No chunks or embeddings provided to upsert_chunks")
        return
        
    if len(chunks) != len(embeddings):
        print(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings")
        return
        
    total_chunks = len(chunks)
    print(f"Starting upload of {total_chunks} chunks in batches of {batch_size}...")
    
    try:
        # Process in batches
        for i in range(0, total_chunks, batch_size):
            batch_end = min(i + batch_size, total_chunks)
            batch_chunks = chunks[i:batch_end]
            batch_embeddings = embeddings[i:batch_end]
            
            # Create vectors for this batch
            vectors = [
                Vector(
                    id=f"{id_prefix}:{i + j}",
                    values=emb,
                    metadata={"text": chunk}
                )
                for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings))
            ]
            
            # Upload batch
            print(f"Uploading batch {i//batch_size + 1}/{(total_chunks + batch_size - 1)//batch_size} "
                  f"(chunks {i+1}-{batch_end} of {total_chunks})...")
            
            try:
                index.upsert(vectors=vectors, namespace=namespace)
                print(f"✅ Successfully uploaded batch of {len(vectors)} vectors")
            except Exception as batch_error:
                print(f"❌ Error uploading batch: {batch_error}")
                # Optionally: Add retry logic here
                raise
                
    except Exception as e:
        print(f"❌ Error in upsert_chunks: {e}")
        raise  # Re-raise to allow the calling code to handle the error


# 3. Query Pinecone
def retrieve_relevant_chunks(query: str, namespace: str, top_k: int = 5) -> list[str]:
    try:
        print(f"\n--- Retrieving chunks for query: '{query}' in namespace: {namespace}")
        
        # Get the embedding for the query
        embedding_response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        )
        embedding = embedding_response.data[0].embedding
        print(f"Generated embedding of length: {len(embedding) if embedding else 0}")

        # Query Pinecone
        print(f"Querying Pinecone index: {index}")
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        print(f"Pinecone query results: {results}")

        matches = getattr(results, "matches", [])
        print(f"Found {len(matches)} matches")
        
        if not matches:
            print("No matches found in Pinecone index")
            return []
            
        # Debug: Print first match metadata keys
        first_match = matches[0]
        print(f"First match metadata keys: {first_match.metadata.keys() if hasattr(first_match, 'metadata') else 'No metadata'}")

        # Safely extract text from metadata
        chunks = []
        for i, match in enumerate(matches):
            if hasattr(match, 'metadata') and match.metadata and 'text' in match.metadata:
                chunks.append(match.metadata["text"])
                if i == 0:  # Print a sample of the first chunk
                    sample = match.metadata["text"][:100] + "..." if len(match.metadata["text"]) > 100 else match.metadata["text"]
                    print(f"Sample chunk: {sample}")
        
        print(f"Returning {len(chunks)} chunks")
        return chunks
        
    except Exception as e:
        print(f"Error in retrieve_relevant_chunks: {str(e)}")
        return []

# 4. Ask ChatGPT with context
def ask_with_context(user_message: str, context_chunks: list[str]) -> str:
    try:
        if not context_chunks:
            return "I couldn't find any relevant information to answer your question."
            
        context_block = "\n---\n".join(context_chunks)
        
        system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
        If the answer cannot be found in the context, say "I don't know" instead of making up an answer."""
        
        user_prompt = f"""Context information is below.
        ---------------------
        {context_block}
        ---------------------
        Given the context information and not prior knowledge, answer the query.
        Query: {user_message}"""
        
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip() or "I don't have enough information to answer that question."
    except Exception as e:
        print("Error in ask_with_context:", e)
        return "I encountered an error while processing your request. Please try again."
    

