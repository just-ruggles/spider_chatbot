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

    .stTextInput > div > div > input {
        background-color: #222 !important;
        color: white !important;
        border-radius: 8px;
        border: 1px solid #666;
    }

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

    .css-1v0mbdj p {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# 🏷️ Título principal
st.title('🕸️ Generación Aumentada por Recuperación (RAG) 💬')
st.write(f"🔍 Versión de Python: {platform.python_version()}")

# 🖼️ Cargar imagen
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"⚠️ No se pudo cargar la imagen: {e}")

# 📌 Sidebar
with st.sidebar:
    st.subheader("🕷️ Los Spider-man somos personas ocupadas y no podemos leer documentos taaaaan largos... necesitamos ayudita analizando las cosas.")

# 🔑 API Key
ke = st.text_input('🔐 Ingresa tu Clave de OpenAI', type="password")
if not ke:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    os.environ['OPENAI_API_KEY'] = ke

# 📄 Cargar PDF
pdf = st.file_uploader("📎 Carga el archivo PDF", type="pdf")

# 🧠 Procesar PDF
if pdf is not None and ke:
    try:
        # Extraer texto
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        
        st.info(f"📃 Texto extraído: {len(text)} caracteres")

        # Dividir texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"🧩 Documento dividido en {len(chunks)} fragmentos")

        # Embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # 💬 Pregunta del usuario
        st.subheader("🤔 ¿Qué quieres saber sobre el documento?")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.markdown("### 🧠 Respuesta:")
            st.markdown(response)

    except Exception as e:
        st.error(f"❌ Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("🔐 Necesitas ingresar tu clave de API de OpenAI para procesar el documento.")
else:
    st.info("📤 Carga un archivo PDF para comenzar.")

