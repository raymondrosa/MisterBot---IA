import os
import streamlit as st
import base64

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# -------------------------
# CONFIGURACIÓN GENERAL
# -------------------------
MODEL_NAME = "llama3"
PDF_PATH = "documento.pdf"
PERSIST_DIR = "memoria/vector_db"
CHAT_MEMORY_FILE = "memoria/chat.txt"
ASSETS_DIR = "assets"

os.makedirs("memoria", exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# -------------------------
# ESTILOS Y FONDO
# -------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_path):
    bin_str = get_base64_of_bin_file(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        .stMarkdown, .stTextInput, h1, h2, h3 {{
            background-color: rgba(255,255,255,0.88);
            padding: 1rem;
            border-radius: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------
# MEMORIA PERSISTENTE
# -------------------------
def cargar_memoria_txt(memory):
    if not os.path.exists(CHAT_MEMORY_FILE):
        return
    with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea.startswith("Usuario:"):
                memory.chat_memory.messages.append(
                    HumanMessage(content=linea.replace("Usuario:", "").strip())
                )
            elif linea.startswith("Asistente:"):
                memory.chat_memory.messages.append(
                    AIMessage(content=linea.replace("Asistente:", "").strip())
                )

def guardar_memoria_txt(pregunta, respuesta):
    with open(CHAT_MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"Usuario: {pregunta}\n")
        f.write(f"Asistente: {respuesta}\n\n")

# -------------------------
# CARGA DEL SISTEMA
# -------------------------
@st.cache_resource
def cargar_sistema():
    llm = ChatOllama(model=MODEL_NAME, temperature=0.2)
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    if os.listdir(PERSIST_DIR):
        db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        texts = splitter.split_documents(docs)

        db = Chroma.from_documents(
            texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        db.persist()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    cargar_memoria_txt(memory)

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        memory=memory
    )

    return qa_chain

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(
    page_title="MisterBot - Asistente Investigador",
    page_icon="🤖",
    layout="wide"
)

# Fondo
fondo_png = os.path.join(ASSETS_DIR, "fondo.png")
fondo_jpg = os.path.join(ASSETS_DIR, "fondo.jpg")
if os.path.exists(fondo_png):
    set_background(fondo_png)
elif os.path.exists(fondo_jpg):
    set_background(fondo_jpg)

# Sidebar
with st.sidebar:
    logo_png = os.path.join(ASSETS_DIR, "logo.png")
    logo_jpg = os.path.join(ASSETS_DIR, "logo.jpg")

    if os.path.exists(logo_png):
        st.image(logo_png, width=200)
    elif os.path.exists(logo_jpg):
        st.image(logo_jpg, width=200)

    st.markdown("---")
    st.metric("Modelo", MODEL_NAME)
    st.metric("Memoria", "Persistente")
    st.metric("Fuente", "PDF + Chat")

    if st.button("🗑️ Limpiar memoria"):
        if os.path.exists(CHAT_MEMORY_FILE):
            os.remove(CHAT_MEMORY_FILE)
        st.rerun()

# Header
col1, col2 = st.columns([1, 6])
with col1:
    if os.path.exists(logo_png):
        st.image(logo_png, width=80)
with col2:
    st.title("🤖 MisterBot")
    st.caption("Asistente investigador con memoria persistente real")

# Sistema
qa_chain = cargar_sistema()

pregunta = st.text_input(
    "💭 Escribe tu pregunta",
    placeholder="Ej: Continúa el análisis que hicimos antes…"
)

if pregunta:
    with st.spinner("🧠 Pensando con memoria y contexto…"):
        result = qa_chain.invoke({"question": pregunta})
        respuesta = result["answer"]

    st.markdown("### 🤖 Respuesta")
    st.write(respuesta)

    guardar_memoria_txt(pregunta, respuesta)

    with st.expander("🧠 Historial de conversación"):
        for msg in qa_chain.memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                st.markdown(f"**👤 Usuario:** {msg.content}")
            else:
                st.markdown(f"**🤖 Asistente:** {msg.content}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;'>"
    "MisterBot; Licencia: cc-nc; 2602194625113; Prof. Raymond Rosa Ávila"
    "</div>",
    unsafe_allow_html=True
)
