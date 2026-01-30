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

st.set_page_config(page_title="SlideSense", page_icon="📘", layout="wide")
load_dotenv()

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def describe_image(image):
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True)

st.markdown("""
<style>
.hero { text-align:center; margin-bottom:25px }
.hero h1 { font-size:44px; font-weight:800 }
.hero p { color:#6b7280; font-size:17px }

.card { background:white; padding:22px; border-radius:16px;
box-shadow:0 10px 25px rgba(0,0,0,0.05) }

.uploader { border:2px dashed #c7d2fe; padding:45px;
border-radius:16px; text-align:center }

.response { background:#f8fafc; padding:20px;
border-radius:14px; margin-top:15px }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>✨ SlideSense</h1>
<p>Chat, Summarize & Analyze PDFs and Images using AI</p>
</div>
""", unsafe_allow_html=True)

mode = st.radio(
    "",
    ["💬 Chat", "📄 Summary", "🖼️ Image"],
    horizontal=True,
    label_visibility="collapsed"
)

left, right = st.columns([1,1])

if mode in ["💬 Chat", "📄 Summary"]:
    with left:
        st.markdown("<div class='card uploader'>Upload PDF</div>", unsafe_allow_html=True)
        pdf = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if pdf:
        with st.spinner("Processing document..."):
            reader = PdfReader(pdf)
            text = ""
            for p in reader.pages:
                text += p.extract_text() + "\n"

            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
            chunks = splitter.split_text(text)

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            vector_db = FAISS.from_texts(chunks, embeddings)

            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        with right:
            if mode == "💬 Chat":
                query = st.text_input(
                    "",
                    placeholder="Ask a question about the document"
                )

                if query:
                    with st.spinner("Generating answer..."):
                        docs = vector_db.similarity_search(query)
                        prompt = ChatPromptTemplate.from_template(
                            "Answer using the context below:\n{context}\nQuestion: {question}"
                        )
                        chain = create_stuff_documents_chain(llm, prompt)
                        answer = chain.invoke(
                            {"context": docs, "question": query}
                        )

                    st.markdown(
                        f"<div class='response'>{answer}</div>",
                        unsafe_allow_html=True
                    )

            if mode == "📄 Summary":
                if st.button("Generate Summary"):
                    with st.spinner("Generating summary..."):
                        full_text = "\n".join(chunks)

                        prompt = ChatPromptTemplate.from_template(
                            """
                            Summarize the document below clearly and concisely.
                            Use bullet points where appropriate.
                            Do not add information not present in the document.

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
