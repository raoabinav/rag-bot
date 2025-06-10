from rag_utils import get_embeddings, upsert_chunks
import os

TRANSCRIPTS_DIR = "transcripts"
NAMESPACE = "avengers-bot"  # your Pinecone namespace

def load_transcript_chunks(file_path: str) -> list[str]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = text.split("\n\n")  # naive paragraph splitter
    return [chunk.strip() for chunk in chunks if chunk.strip()]

if __name__ == "__main__":
    print("🚀 Starting bulk upload...")

    for filename in os.listdir(TRANSCRIPTS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(TRANSCRIPTS_DIR, filename)
            print(f"\n📄 Processing {filename}...")

            chunks = load_transcript_chunks(path)
            print(f"🔹 Found {len(chunks)} chunks.")

            embeddings = get_embeddings(chunks)
            print(f"🔹 Got {len(embeddings)} embeddings.")

            file_id_prefix = filename.replace(".txt", "")
            upsert_chunks(chunks, embeddings, namespace=NAMESPACE, id_prefix=file_id_prefix)

            print(f"✅ Uploaded {filename} to Pinecone.")
