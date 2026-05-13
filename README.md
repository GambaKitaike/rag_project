# HyperMesh / OptiStruct マニュアル RAG デモ

Altair **HyperMesh / OptiStruct** の日本語オンラインマニュアルを対象に、**ハイブリッド検索（ベクトル + BM25）**と **ハイパーリンクに基づく検索結果の拡張**を組み合わせた [RAG](https://en.wikipedia.org/wiki/Retrieval-augmented_generation) アプリです。質問応答は **Streamlit** 上で行います。

**参照マニュアルのトップ URL（例）**  
https://2021.help.altair.com/2021/hwsolvers/ja_jp/os/index.htm

---

## デモ

動画やスクリーンショットをここに置くと、ポートフォリオとして分かりやすくなります。

- （任意）録画 GIF / 動画リンク
- （任意）Streamlit 画面のキャプチャ

過去に掲載した操作デモ（GitHub アセット）:

https://github.com/user-attachments/assets/541d6e79-bc7a-4f95-8f14-961c7642fb81

---

## 概要

- マニュアル HTML を **再帰的に取得**（LangChain `RecursiveUrlLoader` + BeautifulSoup）し、本文とページ間リンクをメタデータに保持します。
- チャンク化したテキストを **Chroma（ベクトル）** と **BM25** の両方で検索し、**EnsembleRetriever** で統合します。
- 各チャンクの `metadata["links"]` を用い、**検索でヒットしたページのリンク先**から追加チャンクを取り込みます（1-hop。本格的な GraphRAG のコミュニティ要約などは含みません）。
- 回答生成には **OpenAI API**（例: `gpt-4o-mini`）を使用します。**埋め込み**は **Hugging Face** の多言語モデル（`intfloat/multilingual-e5-large`）を利用します。

### 著作権・利用について

マニュアル本文の**著作権は Altair 等の権利者に帰属**します。本リポジトリは**マニュアル本文の再配布は行いません**。索引データ（`vectorstore`、`split_documents.pkl` 等）は、各自の環境で `load_data.py` を実行して生成してください。公開サイトの利用条件・robots.txt を遵守してください。

---

## 主な機能

| 機能 | 説明 |
|------|------|
| ハイブリッド検索 | ベクトル類似度（Chroma）と BM25 の加重結合 |
| グラフ拡張検索 | ヒットページの `links` 先からチャンクを追加（Streamlit サイドバーで ON/OFF・上限調整） |
| マニュアル取得 | `RecursiveUrlLoader`（`prevent_outside=True`）、`max_depth` 既定 8（`load_data.py` で変更可） |
| Streamlit UI | 温度・k・ベクトル重み、参照表示件数、グラフ拡張パラメータの調整 |
| タイミングログ | `load_data.py` / ローダーで区間ごとの所要時間を表示 |

---

## アーキテクチャ（概要）

```mermaid
flowchart LR
  crawl[Crawl HTML]
  chunk[Chunk text]
  chroma[Chroma index]
  bm25[BM25 index]
  hybrid[Ensemble retriever]
  graph[Graph expand 1-hop]
  llm[OpenAI Chat]

  crawl --> chunk
  chunk --> chroma
  chunk --> bm25
  chroma --> hybrid
  bm25 --> hybrid
  hybrid --> graph
  graph --> llm
```

---

## 技術スタック

| 区分 | 技術 |
|------|------|
| オーケストレーション | LangChain（`langchain-core` / `langchain-community` / `langchain-chroma` / `langchain-classic` 等） |
| ベクトルストア | Chroma |
| 埋め込み | Hugging Face（`sentence-transformers` / `torch`） |
| スパース検索 | BM25（`rank-bm25`） |
| LLM | OpenAI API（`langchain-openai`） |
| UI | Streamlit |
| HTML 解析 | BeautifulSoup4、`requests` |

---

## ディレクトリ構成（抜粋）

```text
rag_project/
├── README.md
├── requirements.txt
├── .env                    # 各自作成（Git 管理外推奨）
├── data/                   # split_documents.pkl 等（生成物・.gitignore 想定）
├── src/
│   ├── app.py              # Streamlit アプリ
│   ├── load_data.py        # クロール → チャンク → Chroma + pickle 生成
│   ├── data_loader.py      # CLI: クロール結果の pickle + hyperlink JSON
│   ├── data_utils.py       # split 済み Document の保存/読込
│   ├── graph_retrieval.py  # グラフ拡張リトリーバ
│   └── loaders/
│       ├── __init__.py
│       └── altair_manual_loader.py
└── vectorstore/            # Chroma 永続化（生成物・.gitignore 想定）
```

---

## セットアップ

### 前提

- Python 3.11+ 推奨（開発時は 3.14 等でも可。未検証環境は要確認）
- OpenAI API キー（回答生成用）

### インストール

```bash
cd /path/to/rag_project
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate    # macOS / Linux

pip install -r requirements.txt
```

初回は Hugging Face の埋め込みモデル取得で時間がかかることがあります。

### 環境変数

プロジェクトルートに `.env` を作成し、例として次を設定します。

```env
OPENAI_API_KEY=sk-...
```

`load_data.py` / `app.py` は `python-dotenv` で `.env` を読み込みます。

---

## 使い方

作業ディレクトリは **`src`** に合わせると、`vectorstore` の相対パスと `data/split_documents.pkl` のパスがそのまま一致します。

### 1. インデックス構築（初回・マニュアル更新時）

```bash
cd src
python load_data.py
```

- マニュアルの再帰取得・HTML 解析・チャンク分割・埋め込み・Chroma 書き込み・`../data/split_documents.pkl` 保存まで行います。
- `max_depth` は [`load_data.py`](src/load_data.py) 内の `load_altair_manual_documents(..., max_depth=8)` で変更できます（大きいほど取得ページが増え、時間も増えます）。

任意で、クロール結果の pickle とハイパーリンク JSON だけ欲しい場合:

```bash
cd src
python data_loader.py --help
```

### 2. アプリ起動

```bash
cd src
python -m streamlit run app.py
```

Windows で `streamlit` コマンドが PATH にない場合でも、`python -m streamlit` なら動くことが多いです。

### 3. インデックスを作り直したあと

Streamlit の **`@st.cache_resource`** が古いバンドルを掴むことがあります。データ更新後は **アプリの再起動**、または Streamlit メニューから **キャッシュのクリア**を試してください。

---

## 主なパラメータ

| 場所 | 内容 |
|------|------|
| `load_data.py` | `start_url`、`max_depth`、チャンクサイズ / overlap、Embedding モデル名 |
| `app.py` サイドバー | `k`、ベクトル重み、グラフ拡張の有無・追加チャンク上限・リンク先あたりのチャンク数 |

---

## 既知の制限

- `RecursiveUrlLoader` の `prevent_outside=True` により、**クロールの再帰は開始 URL と同一ホスト**に限られます。`metadata["links"]` に含まれる別サブドメインの URL は、リンク情報としては残り得ますが、自動ではクロールされません。
- グラフ拡張は **1-hop のルールベース**であり、学習型の GraphRAG ではありません。

---

## 今後の検討例

- クロスエンコーダによるリランク
- 評価（RAGAS 等）
- Streamlit Community Cloud / Render 等へのデプロイ手順の整理
- より広い Altair 製品ドキュメントへの拡張

---

## ライセンス

本リポジトリの**サンプルコード部分**のライセンスは、リポジトリに `LICENSE` があればそれに従います（未配置の場合は各自の利用方針にご注意ください）。**マニュアル本文の利用**は公式サイトの条件に従ってください。

---

**Last updated:** 2026年5月
