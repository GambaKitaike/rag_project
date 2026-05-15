import os
import pickle
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from data_utils import (
    huggingface_e5_embedding_kwargs,
    resolve_embedding_torch_device,
    save_split_documents,
)
from loaders import load_altair_manual_documents

# USER_AGENT警告対策
os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; MyRAGProject/1.0)"

# ====================== 設定 ======================
BASE_DIR = Path(__file__).parent.resolve()
CRAWLED_PICKLE = BASE_DIR / "../data/crawled_documents.pkl"

start_url = "https://2021.help.altair.com/2021/hwsolvers/ja_jp/os/index.htm"

PERSIST_DIRECTORY = "./vectorstore"

# 出現頻度による共通リンク除去: 全ページ数に対する割合（例: 0.7 なら約 70% 以上のページに出るリンクを links から除外）
# None にすると無効（ナビ除外後の候補をそのまま metadata["links"] に使う）
# 環境変数 RAG_COMMON_LINK_RATIO があれば優先（"off" / "none" / "disable" で無効、0.0～1.0 で比率）
COMMON_LINK_RATIO_THRESHOLD: float | None = 0.7


def _common_link_ratio_threshold() -> float | None:
    raw = os.environ.get("RAG_COMMON_LINK_RATIO")
    if raw is None or str(raw).strip() == "":
        return COMMON_LINK_RATIO_THRESHOLD
    lowered = str(raw).strip().lower()
    if lowered in ("off", "none", "disable"):
        return None
    return float(lowered)


def _fmt_duration(sec: float) -> str:
    if sec >= 3600:
        return f"{int(sec // 3600)}h {int((sec % 3600) // 60)}m {sec % 60:.0f}s"
    if sec >= 60:
        return f"{int(sec // 60)}m {sec % 60:.1f}s"
    return f"{sec:.2f}s"


def _force_crawl_from_env() -> bool:
    v = os.environ.get("RAG_FORCE_CRAWL", "").strip().lower()
    return v in ("1", "true", "yes")


def _load_page_documents() -> list[Document]:
    if not _force_crawl_from_env() and CRAWLED_PICKLE.is_file():
        with CRAWLED_PICKLE.open("rb") as f:
            loaded = pickle.load(f)
        if (
            isinstance(loaded, list)
            and len(loaded) > 0
            and isinstance(loaded[0], Document)
        ):
            print(f"crawled_documents.pkl から読み込み: {CRAWLED_PICKLE} ({len(loaded)} ページ)")
            return loaded
        print("[注意] crawled_documents.pkl が空または Document リストではないため、クロールします。")
    elif _force_crawl_from_env():
        print("RAG_FORCE_CRAWL が有効のため、crawled_documents.pkl は使わずクロールします。")

    print("マニュアルのクロールを開始します...")
    return load_altair_manual_documents(
        start_url,
        max_depth=8,
        common_link_ratio_threshold=_common_link_ratio_threshold(),
    )


def _drop_empty_list_metadata(chunks: list[Document]) -> None:
    """Chroma は空の list メタデータを拒否する。キー削除は graph で欠損=リンクなしとして扱われる。"""
    for doc in chunks:
        md = doc.metadata
        if not md:
            continue
        for key in list(md.keys()):
            val = md.get(key)
            if isinstance(val, list) and len(val) == 0:
                del md[key]


t_run_start = time.perf_counter()

print("マニュアルの読み込みを開始します...")
docs = _load_page_documents()

print(f"✅ 読み込み完了: {len(docs)} ページ")
for i, doc in enumerate(docs):
    print(
        f"  Page {i+1} 文字数: {len(doc.page_content)} "
        f"| links_count={doc.metadata.get('links_count', 0)}"
    )

# ====================== チャンク分割 ======================
t_split_start = time.perf_counter()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "　", " ", ""],
)

chunks = text_splitter.split_documents(docs)
_drop_empty_list_metadata(chunks)
t_split = time.perf_counter() - t_split_start
print(f"[タイミング] チャンク分割: {_fmt_duration(t_split)} → {len(chunks)} チャンク")

# ====================== 無料Embedding & VectorStore ======================
print("Embedding作成 & 保存中...（初回はモデルダウンロードで少し時間がかかります）")

_embed_device = resolve_embedding_torch_device()
print(f"Embedding デバイス: {_embed_device}")

print(
    "Embedding: multilingual-e5-large（query:/passage: プレフィックス使用）。"
    "Chroma を初めて作るか、以前のベクトルと混ぜないよう load_data を実行してください。"
)

t_embed_start = time.perf_counter()
embeddings = HuggingFaceEmbeddings(**huggingface_e5_embedding_kwargs(_embed_device))
t_embed_init = time.perf_counter() - t_embed_start
print(f"[タイミング] Embeddingモデル読み込み: {_fmt_duration(t_embed_init)}")

if os.path.exists(PERSIST_DIRECTORY):
    import shutil
    shutil.rmtree(PERSIST_DIRECTORY)
    print("🗑️  既存vectorstoreを削除しました")

t_chroma_start = time.perf_counter()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)
t_chroma = time.perf_counter() - t_chroma_start
print(
    f"[タイミング] Chroma インデックス作成（全チャンクのベクトル化+保存）: "
    f"{_fmt_duration(t_chroma)}"
)

print(f"完了！ {len(chunks)}個のチャンクを保存しました")
print("無料Embeddingモデルを使用しています")

t_pickle_start = time.perf_counter()
save_split_documents(chunks, "../data/split_documents.pkl")   # プロジェクトルートに保存
t_pickle = time.perf_counter() - t_pickle_start
print(f"[タイミング] split_documents.pkl 保存: {_fmt_duration(t_pickle)}")
print("📦 split_documents.pkl も保存しました（BM25用）")

t_run = time.perf_counter() - t_run_start
print(f"[タイミング] load_data.py 全体: {_fmt_duration(t_run)}")