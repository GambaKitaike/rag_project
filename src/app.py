# src/app.py
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from data_utils import load_split_documents
from graph_retrieval import (
    GraphExpandedRetriever,
    build_adjacency_from_chunks,
    index_chunks_by_url,
)

BASE_DIR = Path(__file__).parent.resolve()

load_dotenv()

# ====================== 設定 ======================
st.set_page_config(page_title="HyperMesh RAG", page_icon="📘", layout="wide")

st.title("📘 HyperMesh / OptiStruct RAG デモ")
st.caption("Altair OptiStruct 2021 マニュアル特化型RAG（無料Embedding使用）")

# サイドバー
with st.sidebar:
    st.header("設定")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1,
                            help="値を大きくすると、回答のランダム性が増します")
    k = st.slider("取得する資料数 (k)", 3, 10, 6)
    weights_vector = st.slider("ベクトル検索の割合(weights)", 0.0, 1.0, 0.65, 0.05)
    display_k = st.slider("参照資料として表示する数", 3, 8, 5,
                          help="多すぎると見づらいので、画面表示はここで制限します")

    st.subheader("ハイパーリンクグラフ拡張")
    use_graph = st.checkbox(
        "グラフで隣接ページのチャンクを追加",
        value=True,
        help="検索でヒットしたページから、マニュアル内リンク先のチャンクを追加取得します。",
    )
    graph_max_extra = st.slider(
        "追加チャンク数の上限",
        0,
        12,
        6,
        help="0 でグラフ拡張なし（ベクトル+BM25のみ）。",
    )
    graph_chunks_per_neighbor = st.slider(
        "各リンク先ページから取るチャンク数",
        1,
        3,
        1,
        help="大きいと文脈は増えますがトークンも増えます。",
    )


@st.cache_resource
def load_search_bundle():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = Chroma(
        persist_directory="./vectorstore",
        embedding_function=embeddings,
    )
    split_path = BASE_DIR / "../data/split_documents.pkl"
    split_docs = load_split_documents(split_path)
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    adjacency = build_adjacency_from_chunks(split_docs)
    chunks_by_url = index_chunks_by_url(split_docs)
    return vectorstore, bm25_retriever, adjacency, chunks_by_url


vectorstore, bm25_retriever, adjacency, chunks_by_url = load_search_bundle()
bm25_retriever.k = k
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

hybrid_core = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[weights_vector, 1.0 - weights_vector],
)

recall_cap = max(12, k * 2)
if use_graph and graph_max_extra > 0:
    hybrid_retriever = GraphExpandedRetriever(
        base_retriever=hybrid_core,
        adjacency=adjacency,
        chunks_by_url=chunks_by_url,
        graph_max_extra=graph_max_extra,
        chunks_per_neighbor=graph_chunks_per_neighbor,
        recall_cap=recall_cap,
    )
else:
    hybrid_retriever = hybrid_core

if use_graph and graph_max_extra > 0 and not adjacency:
    st.sidebar.caption("※ グラフ用の links メタデータがありません。load_data.py で再インデックスしてください。")

# ====================== LLM ======================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=temperature,
)

# ====================== プロンプト ======================
prompt = ChatPromptTemplate.from_template("""
あなたはHyperMeshとOptiStructの専門エンジニアです。
提供された参考資料を基に、正確で実務的な回答をしてください。
資料にないことは「提供された資料には記載がありません」と正直に答えてください。

参考資料:
{context}

質問: {question}
""")


# ====================== Chain ======================
def format_docs(docs):
    return "\n\n".join(
        f"【出典 {i+1}】\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ====================== チャット ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("HyperMesh / OptiStructについて質問してください..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("マニュアルを検索して回答を考えています..."):
            answer = rag_chain.invoke(question)
            st.markdown(answer)

            # 引用資料表示
            with st.expander(f"📑 参照した資料（上位 {display_k} 件を表示）"):
                docs = hybrid_retriever.invoke(question)

                for i, doc in enumerate(docs[:display_k], 1):
                    source = (
                        doc.metadata.get("source_url")
                        or doc.metadata.get("source")
                        or "不明"
                    )
                    page = doc.metadata.get("page", "")
                    page_info = f" (p.{page})" if page else ""

                    st.markdown(f"**[{i}] {source}{page_info}**")

                    preview = doc.page_content[:320]
                    if len(doc.page_content) > 320:
                        preview += "..."
                    st.caption(preview)

                    if i < display_k:
                        st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})
