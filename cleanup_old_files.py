import os
from pathlib import Path

def delete_old_files():
    base_dir = Path(__file__).parent
    
    # Files to delete
    files_to_delete = [
        base_dir / "backend" / "rag_utils.py",
        base_dir / "backend" / "main.py",
        base_dir / "backend" / "config.py",
        base_dir / "cleanup.py",
        base_dir / "setup_dirs.py"
    ]
    
    print("Removing old files:")
    for file_path in files_to_delete:
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"- Deleted: {file_path}")
            except Exception as e:
                print(f"- Error deleting {file_path}: {e}")
        else:
            print(f"- Not found (skipping): {file_path}")
    
    print("\nCleanup complete!")
    print("Current structure:")
    print("""
rag-bot/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   └── chat.py
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   └── chat_service.py
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── openai_utils.py
│   │       └── pinecone_utils.py
│   └── requirements.txt
└── scripts/
    ├── __init__.py
    └── test_refactored_code.py
""")

if __name__ == "__main__":
    delete_old_files()
