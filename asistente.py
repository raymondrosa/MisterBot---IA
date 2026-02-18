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
    """Convierte imagen a base64 para incrustar en CSS"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    """Establece una imagen de fondo para la página"""
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
        margin-bottom: 1rem;
    }}
    .stTextInput > div > div > input {{
        background-color: rgba(255, 255, 255, 0.9);
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 10px;
    }}
    /* Estilo para los mensajes del chat */
    .chat-message {{
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        background-color: rgba(255, 255, 255, 0.95);
        border-left: 4px solid #667eea;
    }}
    .user-message {{
        border-left-color: #38a169;
    }}
    .assistant-message {{
        border-left-color: #667eea;
    }}
    /* Sidebar styling */
    .css-1d391kg {{
        background-color: rgba(255, 255, 255, 0.95);
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def load_css(file_path):
    """Carga un archivo CSS externo"""
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        return True
    return False

def mostrar_logo(logo_path, width=200):
    """Muestra el logo en la barra lateral"""
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=width)
    else:
        st.sidebar.warning("⚠️ Logo no encontrado en assets/")

# -------------------------
# CONFIGURACIÓN GENERAL
# -------------------------
PERSIST_DIR = "memoria/conversaciones"
CHAT_MEMORY_FILE = "memoria/chat.txt"
PDF_PATH = "documento.pdf"
MODEL_NAME = "llama3"
ASSETS_DIR = "assets"

# Crear directorios necesarios
os.makedirs("memoria", exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# -------------------------
# MEMORIA PERSISTENTE TEXTO
# -------------------------
def cargar_memoria_txt(memory):
    """Carga el historial de chat desde archivo"""
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
    """Guarda el historial de chat en archivo"""
    with open(CHAT_MEMORY_FILE, "a", encoding="utf-8") as f:
        f.write(f"Usuario: {pregunta}\n")
        f.write(f"Asistente: {respuesta}\n\n")

# -------------------------
# CARGA DEL SISTEMA
# -------------------------
@st.cache_resource
def cargar_sistema():
    """Carga el modelo, embeddings y base vectorial"""
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
        if not os.path.exists(PDF_PATH):
            st.error(f"❌ No se encuentra el archivo {PDF_PATH}")
            st.stop()
            
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
    page_title="MisterBot - Asistente Investigador",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cargar CSS externo si existe
css_path = os.path.join(ASSETS_DIR, "style.css")
load_css(css_path)

# Aplicar fondo personalizado
fondo_path = os.path.join(ASSETS_DIR, "fondo.jpg")
fondo_png_path = os.path.join(ASSETS_DIR, "fondo.png")

if os.path.exists(fondo_path):
    set_png_as_page_bg(fondo_path)
elif os.path.exists(fondo_png_path):
    set_png_as_page_bg(fondo_png_path)
else:
    # Fondo degradado por defecto
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    </style>
    """, unsafe_allow_html=True)

# BARRA LATERAL
with st.sidebar:
    # Logo en sidebar
    logo_path = os.path.join(ASSETS_DIR, "logo.png")
    logo_jpg_path = os.path.join(ASSETS_DIR, "logo.jpg")
    
    if os.path.exists(logo_path):
        st.image(logo_path, width=200)
    elif os.path.exists(logo_jpg_path):
        st.image(logo_jpg_path, width=200)
    
    st.markdown("---")
    st.markdown("### 📊 Información del Sistema")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Modelo", MODEL_NAME)
    with col2:
        st.metric("Documento", "PDF cargado")
    
    st.markdown("---")
    st.markdown("### 🎯 Características")
    
    features = {
        "✓ Memoria persistente": "💾",
        "✓ Modo investigador": "🔍", 
        "✓ Conversaciones guardadas": "📝",
        "✓ Modelo Llama3": "🦙",
        "✓ Búsqueda semántica": "🎯"
    }
    
    for feature, icon in features.items():
        st.success(f"{icon} {feature}")
    
    st.markdown("---")
    
    # Botón para limpiar memoria
    if st.button("🗑️ Limpiar memoria", use_container_width=True):
        if os.path.exists(CHAT_MEMORY_FILE):
            os.remove(CHAT_MEMORY_FILE)
            st.success("✅ Memoria limpiada correctamente!")
            st.rerun()
    
    # Mostrar estadísticas
    if os.path.exists(CHAT_MEMORY_FILE):
        with open(CHAT_MEMORY_FILE, "r", encoding="utf-8") as f:
            lineas = f.readlines()
            num_mensajes = len([l for l in lineas if l.startswith(("Usuario:", "Asistente:"))])
            st.caption(f"📊 {num_mensajes} mensajes guardados")

# ENCABEZADO PRINCIPAL
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    # Intentar mostrar logo pequeño en encabezado
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
    st.title("🤖 MisterBot")
    st.caption("✨ Memoria persistente real · Modo investigador · Asistente inteligente")

# Cargar el sistema
qa_chain = cargar_sistema()

# INPUT DE PREGUNTA
pregunta = st.text_input(
    "💭 Escribe tu pregunta aquí:", 
    placeholder="Ej: ¿Cuál es el tema principal del documento? ¿Qué conclusiones presenta?",
    key="input_pregunta"
)

if pregunta:
    with st.spinner("🔍 Analizando documento y pensando..."):
        respuesta = qa_chain.invoke({"query": pregunta})
        texto = respuesta["result"]

    # Mostrar respuesta en contenedor con estilo mejorado
    st.markdown("### 🤖 Respuesta")
    
    # Determinar clase CSS para el mensaje
    message_class = "chat-message assistant-message"
    
    st.markdown(
        f'<div class="{message_class}">'
        f'<div style="font-size: 1.1rem; line-height: 1.6;">{texto}</div>'
        f'<div style="font-size: 0.8rem; color: #666; margin-top: 10px;">'
        f'Respondido usando {MODEL_NAME}</div>'
        f'</div>', 
        unsafe_allow_html=True
    )

    # Guardar en memoria
    guardar_memoria_txt(pregunta, texto)

    # Mostrar historial de conversación
    with st.expander("🧠 Memoria de la conversación", expanded=False):
        for i, msg in enumerate(qa_chain.memory.chat_memory.messages):
            rol = "👤 Usuario" if msg.type == "human" else "🤖 Asistente"
            message_class = "user-message" if msg.type == "human" else "assistant-message"
            
            st.markdown(
                f'<div class="chat-message {message_class}">'
                f'<strong>{rol}:</strong><br>{msg.content}'
                f'</div>',
                unsafe_allow_html=True
            )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 20px;'>"
    "🤖 MisterBot - Asistente Investigador con memoria persistente | Creado con LangChain y Streamlit"
    "</div>",
    unsafe_allow_html=True
)