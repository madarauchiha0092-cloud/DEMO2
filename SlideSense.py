import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
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

# -------------------- CSS ANIMATIONS & HOVERS --------------------
st.markdown("""
<style>
.fade-in {
    animation: fadeIn 1.2s ease-in-out;
}
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

.hover-card {
    transition: 0.3s;
    padding: 15px;
    border-radius: 12px;
}
.hover-card:hover {
    transform: scale(1.02);
    box-shadow: 0 0 15px rgba(0,0,0,0.15);
}

button {
    transition: 0.3s !important;
}
button:hover {
    transform: scale(1.05);
    filter: brightness(1.1);
}
</style>
""", unsafe_allow_html=True)

# -------------------- Session State --------------------
defaults = {
    "chat_history": [],
    "vector_db": None,
    "authenticated": False,
    "users": {"admin": "admin123"}
}

for k,v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -------------------- AUTH UI --------------------
def login_ui():
    st.markdown("<h1 class='fade-in' style='text-align:center;'>🔐 SlideSense Login</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # ----- Login -----
    with tab1:
        st.markdown("<div class='hover-card fade-in'>", unsafe_allow_html=True)
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u in st.session_state.users and st.session_state.users[u] == p:
                st.session_state.authenticated = True
                st.success("✅ Login successful")
                st.rerun()
            else:
                st.error("❌ Invalid credentials")
        st.markdown("</div>", unsafe_allow_html=True)

    # ----- Signup -----
    with tab2:
        st.markdown("<div class='hover-card fade-in'>", unsafe_allow_html=True)
        nu = st.text_input("Create Username")
        np = st.text_input("Create Password", type="password")
        if st.button("Sign Up"):
            if nu in st.session_state.users:
                st.warning("User exists")
            elif nu == "" or np == "":
                st.warning("Fields required")
            else:
                st.session_state.users[nu] = np
                st.success("Account created successfully 🎉")
        st.markdown("</div>", unsafe_allow_html=True)

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
    return processor.decode(out[0], skip_special_tokens=True)

# -------------------- AUTH CHECK --------------------
if not st.session_state.authenticated:
    login_ui()
    st.stop()

# -------------------- Sidebar --------------------
st.sidebar.success("✅ Logged in")
if st.sidebar.button("🚪 Logout"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.rerun()

page = st.sidebar.selectbox("Choose Mode", ["PDF Analyzer", "Image Recognition"])

st.sidebar.markdown("## 💬 Chat History")
for i,(q,a) in enumerate(st.session_state.chat_history[-10:],1):
    st.sidebar.markdown(f"Q{i}: {q[:40]}")

# -------------------- HERO --------------------
st.markdown("""
<h1 class='fade-in' style='text-align:center;'>📘 SlideSense</h1>
<p class='fade-in' style='text-align:center;'>AI Powered PDF Analyzer & Image Intelligence System</p>
<hr>
""", unsafe_allow_html=True)

# -------------------- PDF ANALYZER --------------------
if page == "PDF Analyzer":
    st.markdown("<div class='fade-in hover-card'>", unsafe_allow_html=True)
    pdf = st.file_uploader("Upload PDF", type="pdf")
    st.markdown("</div>", unsafe_allow_html=True)

    if pdf:
        if st.session_state.vector_db is None:
            with st.spinner("Processing PDF..."):
                reader = PdfReader(pdf)
                text = ""
                for p in reader.pages:
                    if p.extract_text():
                        text += p.extract_text()+"\n"

                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
                chunks = splitter.split_text(text)

                try:
                    asyncio.get_running_loop()
                except:
                    asyncio.set_event_loop(asyncio.new_event_loop())

                embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
                st.session_state.vector_db = FAISS.from_texts(chunks, embeddings)

        st.success("📄 Document processed successfully")

        q = st.text_input("Ask a question about the document")

        if q:
            docs = st.session_state.vector_db.similarity_search(q, k=5)
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

            history = ""
            for x,y in st.session_state.chat_history[-5:]:
                history += f"Q:{x}\nA:{y}\n"

            prompt = ChatPromptTemplate.from_template("""
History:
{history}

Context:
{context}

Question:
{question}

Rules:
- Answer only from document
- If not found say: Information not found in the document
""")

            chain = create_stuff_documents_chain(llm, prompt)
            res = chain.invoke({"context":docs,"question":q,"history":history})

            st.session_state.chat_history.append((q,res))

        st.markdown("## 💬 Conversation")
        for q,a in st.session_state.chat_history:
            st.markdown(f"<div class='hover-card fade-in'><b>👤 User:</b> {q}<br><b>🤖 SlideSense:</b> {a}</div><br>", unsafe_allow_html=True)

# -------------------- IMAGE RECOGNITION --------------------
if page == "Image Recognition":
    st.markdown("<div class='fade-in hover-card'>", unsafe_allow_html=True)
    img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

    if img_file:
        img = Image.open(img_file)
        st.image(img, use_column_width=True)
        with st.spinner("🔍 Analyzing image..."):
            desc = describe_image(img)
        st.success(desc)
