# 北池 頑張

**概要**  
HyperMesh / OptiStruct（Altair製品）のマニュアル・技術資料に特化したRetrieval-Augmented Generation（RAG）アプリケーションです。

**主な特徴**
- **ハイブリッド検索**（Vector Similarity + BM25）を実装
  - 意味的検索とキーワード検索の両方を組み合わせ、技術用語・コマンド・固有名詞での検索精度を大幅向上
- StreamlitによるインタラクティブUI
- 検索パラメータ（Weights, kなど）をリアルタイム調整可能

**使用技術**
- LangChain
- Chroma (VectorStore)
- OpenAI (Embedding + LLM)
- Streamlit

**今後の展望**
- Reranker（bge-rerankerなど）の追加
- クラウドデプロイ（Streamlit Community Cloud / Render / Hugging Face Spaces）
- 評価指標（RAGASなど）の導入
- より多くのAltair製品対応

**ライブデモ**  

https://github.com/user-attachments/assets/e8afd526-2530-481e-a54d-9d4fb1a93e15


**Last updated:** 2026年4月
