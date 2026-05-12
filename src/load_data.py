import os
import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 無料の日本語対応Embeddingを使う
from langchain_huggingface import HuggingFaceEmbeddings

from data_utils import save_split_documents

# USER_AGENT警告対策
os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; MyRAGProject/1.0)"

# ====================== 設定 ======================
from loaders import load_altair_manual_documents

start_url = "https://2021.help.altair.com/2021/hwsolvers/ja_jp/os/index.htm"

PERSIST_DIRECTORY = "./vectorstore"


def _fmt_duration(sec: float) -> str:
    if sec >= 3600:
        return f"{int(sec // 3600)}h {int((sec % 3600) // 60)}m {sec % 60:.0f}s"
    if sec >= 60:
        return f"{int(sec // 60)}m {sec % 60:.1f}s"
    return f"{sec:.2f}s"


t_run_start = time.perf_counter()

print("マニュアルの読み込みを開始します...")
docs = load_altair_manual_documents(start_url, max_depth=8)

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
t_split = time.perf_counter() - t_split_start
print(f"[タイミング] チャンク分割: {_fmt_duration(t_split)} → {len(chunks)} チャンク")

# ====================== 無料Embedding & VectorStore ======================
print("Embedding作成 & 保存中...（初回はモデルダウンロードで少し時間がかかります）")

t_embed_start = time.perf_counter()
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",   # 日本語・技術文書に強い無料モデル
    # model_name="BAAI/bge-m3",                    # もう一つおすすめ（必要ならこちらに変更）
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
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