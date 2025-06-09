from rag_utils import get_embeddings, upsert_chunks
import os

TRANSCRIPTS_DIR = "transcripts"
NAMESPACE = "avengers-bot"  # or anything else

def load_all_transcripts(folder_path: str) -> list[str]:
    chunks = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            path = os.path.join(folder_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            file_chunks = text.split("\n\n")  # simple paragraph splitter
            chunks.extend(file_chunks)
    return chunks

if __name__ == "__main__":
    chunks = load_all_transcripts(TRANSCRIPTS_DIR)
    embeddings = get_embeddings(chunks)
    upsert_chunks(chunks, embeddings, NAMESPACE)
    print(f"✅ Uploaded {len(chunks)} chunks from {TRANSCRIPTS_DIR} to Pinecone namespace '{NAMESPACE}'")
