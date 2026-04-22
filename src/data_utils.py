# src/data_utils.py
import pickle
from pathlib import Path
from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent.resolve()

def save_split_documents(documents: list[Document], save_path: str = "../data/split_documents.pkl"):
    """split済みDocumentをpickleで保存"""
    path = BASE_DIR / Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "wb") as f:
        pickle.dump(documents, f)
    print(f"✅ {len(documents)}件のsplit済みDocumentを保存しました → {path}")

def load_split_documents(load_path: str = "split_documents.pkl") -> list[Document]:
    """保存したDocumentをロード"""
    path = BASE_DIR / load_path

    with open(path, "rb") as f:
        documents = pickle.load(f)
    print(f"✅ {len(documents)}件のDocumentをロードしました")
    return documents