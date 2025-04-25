import os
import platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

# 🌈 Estilos personalizados
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://pbs.twimg.com/media/F2sr38KWYAAj0bc.jpg:large");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }
    h1 {
        font-size: 48px !important;
        color: #FF4B4B;
        font-weight: bold;
        text-shadow: 2px 2px 8px #000;
    }
    .stTextInput > div > div > input,
    .stTextArea > div > textarea {
        background-color: #222 !important;
        color: white !important;
        border-radius: 8px;
        border: 1px solid #666;
    }
    .stButton > button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stSidebar {
        background-color: rgba(0, 0, 0, 0.6) !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# 🏷️ Título principal
st.title('🕸️ Generación Aumentada por Recuperación (RAG) 💬')
st.write(f"🔍 Versión de Python: {platform.python_version()}")

# 🖼️ Cargar imagen
try:
    image = Image.open('spiderbot.webp')
    st.image(image, width=350)
except FileNotFoundError:
    st.warning("⚠️ Imagen local no encontrada. ¿Seguro que 'spiderbot.webp' está en la misma carpeta?")
except Exception as e:
    st.warning(f"⚠️ No se pudo cargar la imagen: {e}")

# 📌 Sidebar
with st.sidebar:
    st.subheader("🕷️ Los Spider-man estamos ocupados y no podemos leer documentos taaan largos... ¡necesitamos ayuda analítica!")

# 🔑 Clave API
ke = st.text_input('🔐 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("🔐 Necesitas ingresar tu clave API de OpenAI para continuar.")

# 📄 Subir PDF
pdf = st.file_uploader("📎 Carga el archivo PDF", type="pdf")

# 🧠 Procesamiento de PDF
if pdf and ke:
    try:
        # 🔍 Extraer texto
        reader = PdfReader(pdf)
        text = ""
        for page in reader.pages:
            content = page.extract_text()
            if content:
                text += content

        st.info(f"📃 Texto extraído: {len(text)} caracteres")

        # 🧩 Dividir en fragmentos
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = splitter.split_text(text)
        st.success(f"🧩 Documento dividido en {len(chunks)} fragmentos")

        # 🧠 Embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # 🧾 Pregunta del usuario
        st.subheader("🤔 ¿Qué quieres saber del documento?")
        question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        if question:
            # 🔎 Buscar respuesta
            docs = knowledge_base.similarity_search(question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=question)

            st.markdown("### 🧠 Respuesta:")
            st.markdown(response)

    except Exception as e:
        st.error("❌ Error procesando el PDF.")
        import traceback
        st.error(traceback.format_exc())

elif pdf and not ke:
    st.warning("🔐 Debes ingresar tu clave API para procesar el documento.")
else:
    st.info("📥 Sube un archivo PDF para comenzar.")
