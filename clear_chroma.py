import shutil
import os

CHROMA_PATH = "chroma"

def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        print(f"Cleared Chroma database at {CHROMA_PATH}")
    else:
        print("Chroma database directory does not exist.")

clear_database()