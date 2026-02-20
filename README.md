# MisterBot-IA
Licencia: cc-NC; 2602194625113
Desarrollado por Raymond Rosa Ávila; Puerto Rico

🤖 MisterBot – Asistente Investigador con Memoria Persistente

MisterBot es un asistente inteligente desarrollado con Streamlit + LangChain + Ollama + ChromaDB, diseñado para trabajar con:

📄 Documentos PDF como base de conocimiento

🧠 Memoria conversacional persistente real

🎨 Interfaz personalizada con logo y fondo

💾 Vector store persistente

📁 Historial guardado en archivo .txt

🚀 Requisitos del Sistema

Antes de comenzar, asegúrese de tener instalado:

Python 3.10 o superior

Git

Ollama

🧠 1️⃣ Instalar Ollama

Descargar e instalar desde:

👉 https://ollama.com

Luego, descargar el modelo que utiliza MisterBot:

ollama pull llama3

Verificar que funciona:

ollama run llama3

Si responde correctamente, el modelo está listo.

🐍 2️⃣ Crear Entorno Virtual (Recomendado)

En la carpeta del proyecto:

python -m venv venv

Activar el entorno:

Windows

venv\Scripts\activate

Mac / Linux

source venv/bin/activate
📦 3️⃣ Instalar Dependencias

Instalar las librerías necesarias:

pip install streamlit
pip install langchain
pip install langchain-community
pip install chromadb
pip install pypdf
pip install ollama

O si desea crear un archivo requirements.txt:

streamlit
langchain
langchain-community
chromadb
pypdf
ollama

Instalar con:

pip install -r requirements.txt
📁 4️⃣ Estructura del Proyecto

La estructura debe verse así:

MisterBot/
│
├── MisterBot.py
├── documento.pdf
├── assets/
│   ├── logo.png
│   └── fondo.png
│
└── memoria/
    ├── chat.txt
    └── vector_db/

Si la carpeta memoria no existe, el sistema la crea automáticamente.

▶️ 5️⃣ Ejecutar MisterBot

Desde la carpeta del proyecto:

streamlit run MisterBot.py

El navegador abrirá automáticamente en:

http://localhost:8501
🧠 Cómo Funciona la Memoria

MisterBot guarda información en dos niveles:

📄 chat.txt → historial conversacional persistente

🧬 vector_db/ → base de datos vectorial Chroma

Si desea reiniciar memoria:

Presione el botón “Limpiar memoria” en la interfaz
o

Elimine manualmente la carpeta memoria/

🛠 Personalización

Puede modificar en el código:

MODEL_NAME = "llama3"
PDF_PATH = "documento.pdf"

También puede cambiar:

Logo → assets/logo.png

Fondo → assets/fondo.png

🌎 Ejecutarlo en Otra Computadora

Pasos rápidos:

Clonar el repositorio:

git clone https://github.com/USUARIO/MisterBot.git

Entrar al directorio:

cd MisterBot

Instalar dependencias

Instalar Ollama + modelo

Ejecutar con streamlit run MisterBot.py

⚠️ Notas Importantes

Ollama debe estar corriendo en segundo plano.

El modelo debe estar descargado localmente.

El primer arranque puede tardar si crea la base vectorial.

Si cambia el PDF, elimine memoria/vector_db para regenerar embeddings.

📜 Licencia

CC-NC
Desarrollado por Prof. Raymond Rosa Ávila
