import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import asyncio
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# -------------------- Page Configuration --------------------
st.set_page_config(
    page_title="SlideSense",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# -------------------- Session State --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "users" not in st.session_state:
    # default user
    st.session_state.users = {
        "admin": "admin123"
    }

# -------------------- AUTH SYSTEM --------------------
def login_ui():
    st.markdown("<h1 style='text-align:center;'>🔐 SlideSense Login</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username in st.session_state.users and st.session_state.users[username] == password:
                st.session_state.authenticated = True
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")

    with tab2:
        new_user = st.text_input("Create Username")
        new_pass = st.text_input("Create Password", type="password")

        if st.button("Sign Up"):
            if new_user in st.session_state.users:
                st.warning("⚠️ User already exists")
            elif new_user == "" or new_pass == "":
                st.warning("⚠️ Fields cannot be empty")
            else:
                st.session_state.users[new_user] = new_pass
                st.success("✅ Account created successfully")

# -------------------- BLIP Model --------------------
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, blip_model = load_blip()

def describe_image(image: Image.Image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# -------------------- AUTH CHECK --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- Sidebar --------------------
st.sidebar.markdown("## 👤 User Panel")
st.sidebar.success("Logged in")

if st.sidebar.button("🚪 Logout"):
    st.session_state.authenticated = False
    st.session_state.chat_history = []
    st.session_state.vector_db = None
    st.rerun()

page = st.sidebar.selectbox("Choose Mode", ["PDF Analyzer", "Image Recognition"])

st.sidebar.markdown("## 💬 Chat History")
if st.session_state.chat_history:
    for i, (q, a) in enumerate(st.session_state.chat_history[-10:], 1):
        st.sidebar.markdown(f"**Q{i}:** {q[:40]}...")

if st.sidebar.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.sidebar.success("Chat history cleared")

# -------------------- HERO --------------------
st.markdown("""
<h1 style='text-align:center;'>📘 SlideSense</h1>
<p style='text-align:center;'>AI Powered PDF Analyzer & Image Intelligence System</p>
<hr>
""", unsafe_allow_html=True)

# -------------------- PDF ANALYZER --------------------
if page == "PDF Analyzer":
    pdf = st.file_uploader("Upload PDF", type="pdf")

    if pdf:
        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for p in reader.pages:
                    if p.extract_text():
                        text += p.extract_text() + "\n\n"

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                chunks = splitter.split_text(text)

                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.set_event_loop(asyncio.new_event_loop())

                embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        st.success("✅ Document processed successfully")

        user_query = st.text_input("Ask a question about the document")

        if user_query:
            with st.spinner("🤖 Thinking..."):
                docs = st.session_state.vector_db.similarity_search(user_query, k=5)
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

                history_context = ""
                for q, a in st.session_state.chat_history[-5:]:
                    history_context += f"Q: {q}\nA: {a}\n\n"

                prompt = ChatPromptTemplate.from_template(
                    """
You are an AI document assistant.

Conversation History:
{history}

Document Context:
{context}

User Question:
{question}

Rules:
- Answer only from document
- If not found, say: "Information not found in the document"
- Be clear and concise
"""
                )

                chain = create_stuff_documents_chain(llm, prompt)
                response = chain.invoke({
                    "context": docs,
                    "question": user_query,
                    "history": history_context
                })

                st.session_state.chat_history.append((user_query, response))

        st.markdown("## 💬 Conversation")
        for q, a in st.session_state.chat_history:
            st.markdown(f"**👤 User:** {q}")
            st.markdown(f"**🤖 SlideSense:** {a}")
            st.markdown("---")

    else:
        st.info("Upload a PDF to start analysis")

# -------------------- IMAGE RECOGNITION --------------------
if page == "Image Recognition":
    image_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

    if image_file:
        img = Image.open(image_file)
        st.image(img, use_column_width=True)

        with st.spinner("Analyzing image..."):
            desc = describe_image(img)

        st.markdown("## 🖼️ Image Description")
        st.success(desc)

    else:
        st.info("Upload an image for AI analysis")
