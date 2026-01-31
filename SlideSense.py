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

# ---------------- CONFIG ----------------
st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")
load_dotenv()

# ---------------- IMAGE MODEL ----------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

def describe_image(image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

# ---------------- STYLES ----------------
st.markdown("""
<style>
.hero { text-align:center; margin-bottom:30px }
.hero h1 { font-size:44px; font-weight:800 }
.hero p { color:#6b7280; font-size:17px }

.card {
    background:white;
    padding:22px;
    border-radius:16px;
    box-shadow:0 10px 25px rgba(0,0,0,0.05)
}

.uploader {
    border:2px dashed #c7d2fe;
    padding:45px;
    border-radius:16px;
    text-align:center
}

.response {
    background:linear-gradient(145deg,#f8fafc,#eef2ff);
    padding:20px;
    border-radius:14px;
    margin-top:12px;
    border-left:4px solid #6366f1
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
<h1>✨ SlideSense</h1>
<p>Chat, Summarize & Analyze PDFs and Images using AI</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODE SELECT ----------------
mode = st.radio(
    "",
    ["💬 Chat", "📄 Summary", "🖼️ Image"],
    horizontal=True,
    label_visibility="collapsed"
)

left, right = st.columns([1, 1])

# ---------------- SESSION STATE ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

# ---------------- PDF PROCESSING ----------------
@st.cache_resource
def process_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80
    )
    chunks = splitter.split_text(text)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = FAISS.from_texts(chunks, embeddings)

    return vector_db, chunks

# ---------------- PDF MODES ----------------
if mode in ["💬 Chat", "📄 Summary"]:
    with left:
        st.markdown("<div class='card uploader'>Upload PDF</div>", unsafe_allow_html=True)
        pdf = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if pdf:
        with st.spinner("Processing document..."):
            st.session_state.vector_db, st.session_state.chunks = process_pdf(pdf)

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # ---------------- CHAT MODE ----------------
        if mode == "💬 Chat":
            with right:
                # Show chat history
                for msg in st.session_state.messages:
                    st.chat_message(msg["role"]).markdown(msg["content"])

                # Quick actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("📌 Key Points"):
                        st.session_state.messages.append(
                            {"role": "user", "content": "List key points from the document"}
                        )
                        st.rerun()
                with col2:
                    if st.button("🧠 Explain Simply"):
                        st.session_state.messages.append(
                            {"role": "user", "content": "Explain this document in simple terms"}
                        )
                        st.rerun()
                with col3:
                    if st.button("❓ Exam Questions"):
                        st.session_state.messages.append(
                            {"role": "user", "content": "Generate possible exam questions"}
                        )
                        st.rerun()

                # Chat input
                query = st.chat_input("Ask a question about the document")

                if query:
                    st.session_state.messages.append(
                        {"role": "user", "content": query}
                    )

                    with st.spinner("Generating answer..."):
                        docs = st.session_state.vector_db.similarity_search(query)
                        prompt = ChatPromptTemplate.from_template(
                            "Answer using the context below:\n{context}\n\nQuestion: {question}"
                        )
                        chain = create_stuff_documents_chain(llm, prompt)
                        answer = chain.invoke(
                            {"context": docs, "question": query}
                        )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.rerun()

        # ---------------- SUMMARY MODE ----------------
        if mode == "📄 Summary":
            with right:
                if st.button("📄 Generate Summary"):
                    with st.spinner("Generating summary..."):
                        full_text = "\n".join(st.session_state.chunks)
                        prompt = ChatPromptTemplate.from_template(
                            """
                            Summarize the document clearly and concisely.
                            Use bullet points.
                            Do not add external information.

                            Document:
                            {context}
                            """
                        )
                        chain = create_stuff_documents_chain(llm, prompt)
                        summary = chain.invoke({"context": full_text})

                    st.markdown(
                        f"<div class='response'>{summary}</div>",
                        unsafe_allow_html=True
                    )

    else:
        st.info("👈 Upload a PDF to get started")

# ---------------- IMAGE MODE ----------------
if mode == "🖼️ Image":
    with left:
        st.markdown("<div class='card uploader'>Upload Image</div>", unsafe_allow_html=True)
        image_file = st.file_uploader(
            "", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
        )

    if image_file:
        img = Image.open(image_file)
        with right:
            st.image(img, use_column_width=True)
            with st.spinner("Analyzing image..."):
                description = describe_image(img)
            st.markdown(
                f"<div class='response'>{description}</div>",
                unsafe_allow_html=True
            )
            st.caption("🧠 AI-generated image description")
