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
    initial_sidebar_state="collapsed"
)

load_dotenv()

# -------------------- BLIP Model for Image Recognition --------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_image(image: Image.Image):
    """Return text description for uploaded image."""
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)
    return description

# -------------------- Sidebar for Page Selection --------------------
page = st.sidebar.selectbox("Choose Page", ["PDF Analyzer", "Image Recognition"])

# -------------------- CSS Animations & Styling --------------------
st.markdown("""
<style>
/* Your full previous CSS here (hero, glass-card, response-section, uploader, animations, etc.) */
</style>
""", unsafe_allow_html=True)

# -------------------- HERO SECTION --------------------
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">SlideSense</h1>
    <p class="hero-subtitle">Advanced Document & Image Analysis with AI Technology</p>
</div>
""", unsafe_allow_html=True)

# -------------------- PDF Analyzer --------------------
if page == "PDF Analyzer":
    pdf = st.file_uploader("Document Upload", type="pdf", help="Choose a PDF document for analysis")
    
    if pdf is not None:
        with st.spinner('Processing your document...'):
            pdf_reader = PdfReader(pdf)
            text = ""
            for page_pdf in pdf_reader.pages:
                text += page_pdf.extract_text()
                text += "\n\n"

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            splitted_text = text_splitter.split_text(text)

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
            vector_db = FAISS.from_texts(splitted_text, embeddings)

        st.markdown("""
        <div class="success-notification">
            <p class="success-text">✅ Document processed successfully and ready for queries</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="query-section">
            <div class="section-title">Query Interface</div>
            <div class="section-subtitle">Ask intelligent questions about your document</div>
        </div>
        """, unsafe_allow_html=True)

        user_query = st.text_input(
            "Ask a question",
            placeholder="Enter your question about the document...",
            help="Type your question here",
            label_visibility="collapsed"
        )

        if user_query:
            with st.spinner('🤖 Generating intelligent response...'):
                docs = vector_db.similarity_search(user_query)
                llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
                prompt = ChatPromptTemplate.from_template(
                    "Answer the following:\n{context}\nQuestion: {question}\n Read the context carefully and then answer it"
                )
                chain = create_stuff_documents_chain(llm, prompt)
                response = chain.invoke({"context": docs, "question": user_query})

            st.markdown(f"""
            <div class="response-section">
                <div class="response-header">
                    <span class="response-icon">🤖</span>
                    <h3 class="response-title">Response</h3>
                </div>
                <div class="response-content">{response}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="instruction-container">
            <div class="instruction-title">Get Started</div>
            <div class="instruction-text">
                Upload your PDF document above to unlock the power of AI-driven document analysis. 
                Ask questions, extract insights, and discover information with intelligent search capabilities.
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------- IMAGE RECOGNITION --------------------
if page == "Image Recognition":
    image_file = st.file_uploader("Upload an Image", type=["png","jpg","jpeg"], help="Choose an image for recognition")
    
    if image_file is not None:
        img = Image.open(image_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner("🔍 Analyzing image..."):
            description = describe_image(img)
        
        st.markdown(f"""
        <div class="response-section">
            <div class="response-header">
                <span class="response-icon">🖼️</span>
                <h3 class="response-title">Image Description</h3>
            </div>
            <div class="response-content">{description}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="instruction-container">
            <div class="instruction-title">Get Started</div>
            <div class="instruction-text">
                Upload an image to get an AI-generated description of its contents.
            </div>
        </div>
        """, unsafe_allow_html=True)
