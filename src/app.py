# src/app.py
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# ====================== 設定 ======================
st.set_page_config(page_title="HyperMesh RAG", page_icon="📘", layout="wide")

st.title("📘 HyperMesh / OptiStruct RAG デモ")
st.caption("Altair OptiStruct 2021 マニュアル特化型RAG（無料Embedding使用）")

# サイドバー
with st.sidebar:
    st.header("設定")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.3, 0.1)
    k = st.slider("取得する資料数 (k)", 3, 10, 6)

# ====================== ベクトルDBロード ======================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
        model_kwargs={'device': 'cpu'},   # GPUを使いたい場合は 'cuda' に変更
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = Chroma(
        persist_directory="./vectorstore",
        embedding_function=embeddings
    )
    return vectorstore

vectorstore = load_vectorstore()
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": k})

# ====================== LLM ======================
from langchain_openai import ChatOpenAI   # まだOpenAIのLLMは使えるはず

llm = ChatOpenAI(
    model="gpt-4o-mini",      # 回答生成だけなら無料枠が残っていれば動く
    temperature=temperature
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
    {"context": vector_retriever | format_docs, "question": RunnablePassthrough()}
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
            with st.expander("📑 参照した資料"):
                docs = vector_retriever.invoke(question)
                for i, doc in enumerate(docs, 1):
                    st.markdown(f"**[{i}]** {doc.metadata.get('source', '不明')}")
                    preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                    st.caption(preview)

    st.session_state.messages.append({"role": "assistant", "content": answer})