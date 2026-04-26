**概要**  
HyperMesh / OptiStruct（Altair製品）のマニュアル・技術資料に特化したRetrieval-Augmented Generation（RAG）アプリケーションです。  
マニュアルは下記URLから一部抜粋して使用しています。  
- https://2021.help.altair.com/2021/hwsolvers/ja_jp/os/index.htm

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
- Hyperlink Graphを作成し、HTMLファイルに特化した検索アルゴリズムを実装
- Reranker（bge-rerankerなど）の追加
- クラウドデプロイ（Streamlit Community Cloud / Render / Hugging Face Spaces）
- 評価指標（RAGASなど）の導入
- より多くのAltair製品対応

**ライブデモ**  

https://github.com/user-attachments/assets/541d6e79-bc7a-4f95-8f14-961c7642fb81

**Last updated:** 2026年4月
