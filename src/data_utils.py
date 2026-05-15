# src/data_utils.py
import os
import pickle
import re
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

BASE_DIR = Path(__file__).parent.resolve()

# intfloat/multilingual-e5-large は検索用途で query:/passage: プレフィックスが推奨される。
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

_CJK_CHAR = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")


def bm25_tokenize_japanese(text: str) -> list[str]:
    """日本語向け BM25 用トークン。`text.split()` だけだとスペース無し質問が1トークンになり検索不能になる。"""
    if not text:
        return []
    flat = text.replace("\n", " ").strip()
    if not flat:
        return []
    tokens: list[str] = []
    for part in flat.split():
        p = part.strip()
        if not p:
            continue
        if p.isascii() and re.fullmatch(r"[A-Za-z0-9_]+", p):
            tokens.append(p.lower())
        elif _CJK_CHAR.search(p):
            tokens.extend(
                c
                for c in p
                if (_CJK_CHAR.fullmatch(c) is not None) or c.isdigit()
            )
        else:
            tokens.append(p.lower())
    return tokens or [flat.lower()]


def huggingface_e5_embedding_kwargs(device: str) -> dict[str, Any]:
    """Chroma・検索で共通の multilingual-e5-large 設定（非対称検索用プレフィックス付き）。"""
    return {
        "model_name": EMBEDDING_MODEL_NAME,
        "model_kwargs": {"device": device},
        "encode_kwargs": {"normalize_embeddings": True, "prompt": "passage: "},
        "query_encode_kwargs": {"normalize_embeddings": True, "prompt": "query: "},
    }


def _cuda_kernels_runnable() -> bool:
    """ドライバは通るが wheel に当該 GPU 向け SASS が無い場合の no kernel image を事前検出する。"""
    import torch

    if not torch.cuda.is_available():
        return False
    try:
        x = torch.tensor([1.0, 2.0], device="cuda")
        y = x * x + 1.0
        _ = float(y.sum().item())
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def resolve_embedding_torch_device() -> str:
    """SentenceTransformer / HuggingFaceEmbeddings 用の torch デバイス。CUDA → MPS → CPU。

    環境変数 RAG_EMBEDDING_DEVICE に cpu / cuda / mps を指定するとそれを優先
    （cuda 指定時も、実機でカーネルが実行できなければ cpu に落とす）。
    """
    override = os.environ.get("RAG_EMBEDDING_DEVICE", "").strip().lower()
    if override == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        return "cpu"

    cuda_ok = torch.cuda.is_available() and _cuda_kernels_runnable()

    if override == "cuda":
        if cuda_ok:
            return "cuda"
        print(
            "[注意] RAG_EMBEDDING_DEVICE=cuda ですが GPU でカーネルが実行できません。"
            "古い GPU と PyTorch+cu126 wheel の組み合わせで起きることがあります。CPU にフォールバックします。"
        )
        return "cpu"

    if override == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    # 自動: CUDA が見えてもカーネル非対応なら CPU
    if torch.cuda.is_available() and not cuda_ok:
        print(
            "[注意] CUDA は利用可能ですが、この GPU 向けのビルド済みカーネルがありません"
            "（cudaErrorNoKernelImageForDevice）。CPU にフォールバックします。"
        )
    if cuda_ok:
        return "cuda"

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


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