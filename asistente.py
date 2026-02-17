import os
import streamlit as st

from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory

# -------------------------
# CONFIGURACIÓN GENERAL
# -------------------------
PERSIST_DIR = "memoria/conversaciones"
CHAT_MEMORY_FILE = "memoria/chat.txt"
PDF_PATH = "documento.pdf"
MODEL_NAME = "llama3"

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

    # 4. Memoria conversacional (RAM + TXT)
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
    layout="wide"
)

st.title("🤖 MisterBot")
st.caption("Memoria persistente real · Modo investigador")

qa_chain = cargar_sistema()

pregunta = st.text_input("🤖 Dime:")

if pregunta:
    with st.spinner("Pensando..."):
        respuesta = qa_chain.invoke({"query": pregunta})
        texto = respuesta["result"]

    st.markdown("### 🤖 Respuesta")
    st.write(texto)

    guardar_memoria_txt(pregunta, texto)

    with st.expander("🧠 Memoria de la conversación"):
        for msg in qa_chain.memory.chat_memory.messages:
            rol = "👤 Usuario" if msg.type == "human" else "🤖 Asistente"
            st.markdown(f"**{rol}:** {msg.content}")
