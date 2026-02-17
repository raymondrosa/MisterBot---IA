import os
import streamlit as st

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# -------------------------
# CONFIGURACIÓN GENERAL
# -------------------------
PERSIST_DIR = "memoria/conversaciones"
PDF_PATH = "documento.pdf"   # cambia esto a tu PDF real
MODEL_NAME = "llama3"

# -------------------------
# CARGA DEL SISTEMA
# -------------------------
@st.cache_resource
def cargar_sistema():

    # 1. Modelo
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.2
    )

    # 2. Embeddings
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # 3. Si ya existe memoria, la cargamos
    if os.path.exists(PERSIST_DIR):
        db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        # 4. Cargar PDF solo la primera vez
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        texts = splitter.split_documents(docs)

        db = Chroma.from_documents(
            texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        db.persist()

    # 5. Memoria conversacional
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 6. Cadena QA con memoria
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        memory=memory,
        return_source_documents=False
    )

    return qa


# -------------------------
# INTERFAZ STREAMLIT
# -------------------------
st.set_page_config(page_title="Asistente Investigador", layout="wide")

st.title("🧠 Asistente Investigador")
st.caption("Memoria persistente activada · Modo análisis")

qa_chain = cargar_sistema()

pregunta = st.text_input("🧠 Escribe tu pregunta:")

if pregunta:
    with st.spinner("Pensando..."):
        respuesta = qa_chain.invoke({"query": pregunta})

    st.markdown("### 🤖 Respuesta")
    st.write(respuesta["result"])

    with st.expander("🧠 Memoria de la conversación"):
        for msg in qa_chain.memory.chat_memory.messages:
            rol = "👤 Usuario" if msg.type == "human" else "🤖 Asistente"
            st.markdown(f"**{rol}:** {msg.content}")
