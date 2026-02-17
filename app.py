import streamlit as st
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

# =========================
# CONFIGURACIÓN GENERAL
# =========================

PDF_PATH = "documento.pdf"   # 👈 asegúrate que exista
LLM_MODEL = "llama3"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = "chroma_db"

# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="Asistente Investigador",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Asistente Investigador")
st.caption("Respuestas en tiempo real · Modo análisis activado")

# =========================
# CARGA DEL SISTEMA (UNA VEZ)
# =========================

def cargar_sistema():
    with st.spinner("📄 Cargando documento..."):
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
        st.success(f"Documento cargado: {len(docs)} páginas")

    with st.spinner("✂️ Fragmentando conocimiento..."):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150
        )
        texts = splitter.split_documents(docs)
        st.success(f"Fragmentos creados: {len(texts)}")

    with st.spinner("🧠 Inicializando embeddings (local)..."):
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    with st.spinner("📦 Construyendo memoria vectorial..."):
        db = Chroma.from_documents(
            texts,
            embedding=embeddings,
            persist_directory=CHROMA_DIR
        )

    with st.spinner("🤖 Despertando el modelo cognitivo..."):
        llm = Ollama(
            model=LLM_MODEL,
            temperature=0.2
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(),
            return_source_documents=False
        )

    return qa

# =========================
# INICIALIZACIÓN CONTROLADA
# =========================

if "qa" not in st.session_state:
    st.info("Inicializando sistema por primera vez…")
    st.session_state.qa = cargar_sistema()
    st.success("Sistema listo. Puedes preguntar.")

qa_chain = st.session_state.qa

# =========================
# INTERACCIÓN CON EL USUARIO
# =========================

st.divider()

pregunta = st.text_input(
    "🧠 Haz tu pregunta:",
    placeholder="Ej. ¿Cuál es la idea principal del documento?"
)

if pregunta:
    with st.spinner("✍️ Pensando…"):
        inicio = time.time()
        respuesta = qa_chain.invoke(pregunta)
        fin = time.time()

    st.markdown("### 📝 Respuesta")
    st.write(respuesta["result"])

    st.caption(f"⏱️ Tiempo de respuesta: {fin - inicio:.2f} segundos")
