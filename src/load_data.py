# src/load_data.py
import bs4
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# 無料の日本語対応Embeddingを使う
from langchain_huggingface import HuggingFaceEmbeddings



load_dotenv()

# USER_AGENT警告対策
os.environ["USER_AGENT"] = "Mozilla/5.0 (compatible; MyRAGProject/1.0)"

# ====================== 設定 ======================
from crawler import PoliteCrawler

start_url = "https://2021.help.altair.com/2021/hwsolvers/ja_jp/os/index.htm"
crawler = PoliteCrawler(start_url, max_depth=5, save_dir="../data")
urls = crawler.get_all_urls()

PERSIST_DIRECTORY = "./vectorstore"

print("マニュアルの読み込みを開始します...")

loader = WebBaseLoader(
    web_paths=urls,
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(name=["div", "p", "h1", "h2", "h3", "h4", "li", "pre", "span"])
    ),
)

docs = loader.load()

print(f"✅ 読み込み完了: {len(docs)} ページ")
for i, doc in enumerate(docs):
    print(f"  Page {i+1} 文字数: {len(doc.page_content)}")

# ====================== チャンク分割 ======================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", "！", "？", "　", " ", ""],
)

chunks = text_splitter.split_documents(docs)
print(f"チャンク分割完了: {len(chunks)} 個")

# ====================== 無料Embedding & VectorStore ======================
print("Embedding作成 & 保存中...（初回はモデルダウンロードで少し時間がかかります）")

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",   # 日本語・技術文書に強い無料モデル
    # model_name="BAAI/bge-m3",                    # もう一つおすすめ（必要ならこちらに変更）
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

if os.path.exists(PERSIST_DIRECTORY):
    import shutil
    shutil.rmtree(PERSIST_DIRECTORY)
    print("🗑️  既存vectorstoreを削除しました")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)

print(f"完了！ {len(chunks)}個のチャンクを保存しました")
print("無料Embeddingモデルを使用しています")