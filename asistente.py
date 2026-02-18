import os
import streamlit as st
import base64
from pathlib import Path

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# -------------------------
# FUNCIONES PARA ESTILOS
# -------------------------
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    /* Hacer el contenido legible sobre el fondo */
    .stMarkdown, .stTextInput, .stCaption, h1, h2, h3 {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }}
    .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.9);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def mostrar_logo(logo_path, width=200):
    """Muestra el logo en la barra lateral o encabezado"""
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=width)
        # También puedes ponerlo en el encabezado principal:
        # st.image(logo_path, width=width)
    else:
        st.sidebar.warning("Logo no encontrado")

# -------------------------
# CONFIGURACIÓN GENERAL
# -------------------------
PERSIST_DIR = "memoria/conversaciones"
CHAT_MEMORY_FILE = "memoria/chat.txt"
PDF_PATH = "documento.pdf"
MODEL_NAME = "llama3"
ASSETS_DIR = "assets"  # Carpeta donde guardas imágenes

os.makedirs("memoria", exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

# -------------------------
# MEMORIA PERSISTENTE TEXTO
# -------------------------
def cargar_memoria_txt(memory):
    if not os.path.exists(CHAT_MEMORY_FILE):
        return

    with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea.startswith("Usuario:"):
                memory.chat_memory.add_user_message(
                    linea.replace("Usuario:", "").strip()
                )
            elif linea.startswith("Asistente:"):
                memory.chat_memory.add_ai_message(
                    linea.replace("Asistente:", "").strip()
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
    # 1. Modelo
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.2
    )

    # 2. Embeddings
    embeddings = OllamaEmbeddings(model=MODEL_NAME)

    # 3. Base vectorial (PDF)
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
    else:
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

    # 4. Memoria conversacional
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    cargar_memoria_txt(memory)

    # 5. Cadena QA
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=False
    )

    return qa

# -------------------------
# INTERFAZ STREAMLIT
# -------------------------
st.set_page_config(
    page_title="Asistente Investigador",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# APLICAR FONDO (si existe)
fondo_path = os.path.join(ASSETS_DIR, "fondo.jpg")  # o fondo.png
if os.path.exists(fondo_path):
    set_png_as_page_bg(fondo_path)
else:
    # Fondo por defecto si no hay imagen
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# MOSTRAR LOGO EN SIDEBAR
logo_path = os.path.join(ASSETS_DIR, "logo.png")
mostrar_logo(logo_path, width=180)

# BARRA LATERAL CON INFORMACIÓN
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 Información del Sistema")
    st.info(f"Modelo: {MODEL_NAME}")
    st.info(f"Documento: {os.path.basename(PDF_PATH)}")
    
    st.markdown("---")
    st.markdown("### 🎯 Características")
    st.success("✓ Memoria persistente")
    st.success("✓ Modo investigador")
    st.success("✓ Conversaciones guardadas")
    
    # Botón para limpiar memoria (opcional)
    if st.button("🗑️ Limpiar memoria"):
        if os.path.exists(CHAT_MEMORY_FILE):
            os.remove(CHAT_MEMORY_FILE)
            st.success("Memoria limpiada!")
            st.rerun()

# ENCABEZADO PRINCIPAL CON LOGO PEQUEÑO
col1, col2 = st.columns([1, 5])
with col1:
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
with col2:
    st.title("🤖 MisterBot")
    st.caption("Memoria persistente real · Modo investigador")

qa_chain = cargar_sistema()

# INPUT DE PREGUNTA
pregunta = st.text_input("💭 Escribe tu pregunta aquí:", placeholder="Ej: ¿Cuál es el tema principal del documento?")

if pregunta:
    with st.spinner("🔍 Pensando..."):
        respuesta = qa_chain.invoke({"query": pregunta})
        texto = respuesta["result"]

    # Mostrar respuesta en contenedor con estilo
    with st.container():
        st.markdown("### 🤖 Respuesta")
        st.markdown(f'<div style="background-color: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 10px;">{texto}</div>', 
                   unsafe_allow_html=True)

    guardar_memoria_txt(pregunta, texto)

    with st.expander("🧠 Memoria de la conversación"):
        for msg in qa_chain.memory.chat_memory.messages:
            rol = "👤 Usuario" if msg.type == "human" else "🤖 Asistente"
            st.markdown(f"**{rol}:** {msg.content}")