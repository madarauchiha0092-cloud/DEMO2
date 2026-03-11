import streamlit as st
try:
    from streamlit_lottie import st_lottie
    LOTTIE_AVAILABLE = True
except ImportError:
    LOTTIE_AVAILABLE = False
    def st_lottie(*args, **kwargs): pass

from datetime import datetime
import uuid
import requests
import time
import json
from PyPDF2 import PdfReader
from PIL import Image
import base64
import os
import streamlit.components.v1 as components

import firebase_admin
from firebase_admin import credentials, auth, firestore
from google.api_core.exceptions import ResourceExhausted

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


st.set_page_config(page_title="SlideSense AI", page_icon="🧠", layout="wide")


# -------------------- LOGO --------------------
def get_logo_base64():
    logo_path = "assets/logo.png"
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None

LOGO_B64 = get_logo_base64()

def logo_img_tag(size=52):
    if LOGO_B64:
        return f'<img src="data:image/png;base64,{LOGO_B64}" width="{size}" height="{size}" style="border-radius:10px;object-fit:contain;">'
    return f'<span style="font-size:{size//2}px;">🧠</span>'


# -------------------- CSS --------------------
st.markdown("""
<style>
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}

.logo-bar {display:flex;align-items:center;gap:14px;padding:18px 0 10px 0;}
.logo-text {
    font-size:2.2rem;font-weight:900;letter-spacing:5px;
    background:linear-gradient(135deg,#6C63FF 0%,#48CAE4 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    background-clip:text;margin:0;line-height:1.1;
}
.logo-tagline {font-size:0.7rem;color:#888;letter-spacing:2.5px;text-transform:uppercase;margin:0;}

.sidebar-logo {display:flex;align-items:center;gap:10px;padding:8px 0 14px 0;
    border-bottom:1px solid rgba(108,99,255,0.2);margin-bottom:12px;}
.sidebar-logo-text {font-size:1rem;font-weight:800;letter-spacing:3px;
    background:linear-gradient(135deg,#6C63FF 0%,#48CAE4 100%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
</style>
""", unsafe_allow_html=True)


# -------------------- FIREBASE --------------------
if not firebase_admin._apps:
    raw_key = st.secrets["firebase"]["private_key"]
    private_key = raw_key.replace("\\n", "\n")
    firebase_config = {
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": private_key,
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
    }
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# -------------------- SESSION --------------------
defaults = {
    "authenticated": False,
    "is_guest": False,
    "user_id": None,
    "email": None,
    "mode": "PDF",
    "current_chat_id": None,
    "vector_db": None,
    "guest_messages": [],
    "logo_typed": False,
    "response_cache": {},   # {question_hash: answer} — avoids duplicate API calls
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# -------------------- HELPERS --------------------
def load_lottie(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def type_text_logo():
    if st.session_state.logo_typed:
        st.markdown(f"""
            <div class="logo-bar">
                {logo_img_tag(56)}
                <div>
                    <p class="logo-text">SLIDESENSE</p>
                    <p class="logo-tagline">AI · PDF · Image Analyzer</p>
                </div>
            </div>""", unsafe_allow_html=True)
        return
    placeholder = st.empty()
    full = "SLIDESENSE"
    out = ""
    for c in full:
        out += c
        placeholder.markdown(f"""
            <div class="logo-bar">
                {logo_img_tag(56)}
                <div>
                    <p class="logo-text">{out}</p>
                    <p class="logo-tagline">AI · PDF · Image Analyzer</p>
                </div>
            </div>""", unsafe_allow_html=True)
        time.sleep(0.07)
    st.session_state.logo_typed = True


def render_logo():
    st.markdown(f"""
        <div class="logo-bar">
            {logo_img_tag(56)}
            <div>
                <p class="logo-text">SLIDESENSE</p>
                <p class="logo-tagline">AI · PDF · Image Analyzer</p>
            </div>
        </div>""", unsafe_allow_html=True)


def render_answer_with_copy(answer: str, accuracy_line: str = "", accuracy_color: str = "") -> None:
    st.markdown(answer)
    safe_text = json.dumps(answer)

    # Build accuracy badge HTML (only for PDF mode when accuracy exists)
    if accuracy_line and accuracy_color:
        acc_html = f"""
        <div style="
            margin-bottom:10px;
            padding:8px 14px;
            border-radius:8px;
            background:rgba(0,0,0,0.05);
            border-left:4px solid {accuracy_color};
            display:inline-block;
            font-size:0.85rem;
            font-weight:600;
            color:{accuracy_color};
        ">
            📊 {accuracy_line}
        </div>
        <br>"""
        extra_height = 55
    else:
        acc_html = ""
        extra_height = 0

    components.html(f"""
        {acc_html}
        <script>
        function copyText() {{
            var text = {safe_text};
            try {{
                var el = document.createElement('textarea');
                el.value = text;
                el.setAttribute('readonly','');
                el.style.position='absolute';
                el.style.left='-9999px';
                document.body.appendChild(el);
                el.select();
                document.execCommand('copy');
                document.body.removeChild(el);
                var btn = document.getElementById('copybtn');
                btn.innerText = '✅ Copied!';
                btn.style.borderColor='#48CAE4';
                btn.style.color='#48CAE4';
                setTimeout(function(){{
                    btn.innerText='📋 Copy';
                    btn.style.borderColor='#6C63FF';
                    btn.style.color='#6C63FF';
                }}, 2000);
            }} catch(e) {{ window.prompt("Copy:", text); }}
        }}
        </script>
        <button id="copybtn" onclick="copyText();"
            style="margin-top:4px;padding:5px 14px;border-radius:6px;
                   border:1px solid #6C63FF;color:#6C63FF;
                   background:transparent;cursor:pointer;font-size:12px;">
            📋 Copy
        </button>""", height=45 + extra_height)


# -------------------- AUTH --------------------
def signup(email, password):
    try:
        return auth.create_user(email=email, password=password)
    except auth.EmailAlreadyExistsError:
        st.error("An account with this email already exists.")
        return None
    except Exception as e:
        st.error(f"Signup error: {e}")
        return None


def login(email, password):
    try:
        if "FIREBASE_WEB_API_KEY" not in st.secrets:
            st.error("FIREBASE_WEB_API_KEY missing from secrets.")
            return None
        api_key = st.secrets["FIREBASE_WEB_API_KEY"]
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
        data = requests.post(url, json={"email": email, "password": password, "returnSecureToken": True}).json()
        if "error" in data:
            msg = data["error"]["message"]
            st.error("❌ Invalid email or password." if msg in ("EMAIL_NOT_FOUND","INVALID_LOGIN_CREDENTIALS","INVALID_PASSWORD") else f"Login failed: {msg}")
            return None
        return auth.get_user_by_email(email)
    except Exception as e:
        st.error(f"Login error: {e}")
        return None


def reset_password(email):
    try:
        if "FIREBASE_WEB_API_KEY" not in st.secrets:
            return False
        api_key = st.secrets["FIREBASE_WEB_API_KEY"]
        url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={api_key}"
        data = requests.post(url, json={"requestType": "PASSWORD_RESET", "email": email}).json()
        if "error" in data:
            st.error(f"Reset failed: {data['error']['message']}")
            return False
        return True
    except Exception as e:
        st.error(f"Reset error: {e}")
        return False


# -------------------- FIRESTORE --------------------
def get_next_chat_number(user_id, mode):
    chats = db.collection("users").document(user_id).collection("chats").where("mode","==",mode).stream()
    return sum(1 for _ in chats) + 1


def create_new_chat(user_id, mode):
    chat_id = str(uuid.uuid4())
    num = get_next_chat_number(user_id, mode)
    icon = "📘" if mode == "PDF" else "🖼"
    title = f"{icon} Chat {num}"
    db.collection("users").document(user_id).collection("chats").document(chat_id).set({
        "mode": mode, "created_at": datetime.utcnow(), "title": title
    })
    return chat_id


def update_chat_title(user_id, chat_id, first_question):
    short = first_question.strip()[:35]
    if len(first_question.strip()) > 35:
        short += "..."
    db.collection("users").document(user_id).collection("chats").document(chat_id).update({"title": short})


def save_message(user_id, chat_id, role, content):
    db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages").add({
            "role": role, "content": content, "timestamp": datetime.utcnow()
        })


def load_user_chats(user_id, mode):
    chats = db.collection("users").document(user_id).collection("chats") \
        .where("mode","==",mode) \
        .order_by("created_at", direction=firestore.Query.DESCENDING).stream()
    return [(doc.id, doc.to_dict()["title"]) for doc in chats]


def load_messages(user_id, chat_id):
    msgs = db.collection("users").document(user_id).collection("chats") \
        .document(chat_id).collection("messages") \
        .order_by("timestamp").stream()
    return [(doc.to_dict()["role"], doc.to_dict()["content"]) for doc in msgs]


# -------------------- LLM --------------------
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3,
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        max_retries=2,
        max_output_tokens=1024,
    )

def call_llm_with_retry(fn, *args, max_attempts=5, **kwargs):
    """Exponential backoff retry — silently waits on quota errors, never crashes."""
    import time as _time
    delays = [5, 15, 30, 60, 90]
    last_err = None
    for attempt, delay in enumerate(delays[:max_attempts], 1):
        try:
            result = fn(*args, **kwargs)
            return result, None
        except ResourceExhausted:
            last_err = "quota"
            if attempt < max_attempts:
                wait_slot = st.empty()
                for remaining in range(delay, 0, -1):
                    wait_slot.info(
                        f"⏳ API rate limit hit — auto-retrying in **{remaining}s** "
                        f"(attempt {attempt}/{max_attempts})..."
                    )
                    _time.sleep(1)
                wait_slot.empty()
        except Exception as _ex:
            last_err = type(_ex).__name__
            if attempt < max_attempts:
                _time.sleep(5)
    if last_err == "quota":
        return None, (
            "⚠️ **Gemini API quota exhausted after multiple retries.**\n\n"
            "The free tier allows ~15 requests/min. Please wait 1–2 minutes, "
            "then ask your question again."
        )
    return None, f"⚠️ **Error ({last_err})** — please try again in a moment."


# -------------------- PDF HELPERS --------------------
@st.cache_resource(show_spinner=False)
def get_embeddings_model():
    """Load once, reuse forever — avoids reload on every PDF upload."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )


def extract_pdf_text_safe(pdf_file):
    """
    Extract text page-by-page with full error isolation.
    Returns (text, page_count).  Never raises.
    """
    try:
        reader = PdfReader(pdf_file)
    except Exception as e:
        return "", 0, f"Cannot open PDF: {e}"

    pages_text = []
    total = len(reader.pages)
    failed = 0
    for page in reader.pages:
        try:
            pages_text.append(page.extract_text() or "")
        except Exception:
            pages_text.append("")
            failed += 1

    combined = "\n".join(pages_text).strip()
    warn = f" ({failed} pages could not be read)" if failed else ""
    return combined, total, warn


def build_vector_db(text, total_pages, progress_cb=None):
    """
    True incremental batched embedding into FAISS.
    Each batch is embedded and merged immediately — no single blocking call.
    Caps at 400 chunks max for speed and memory safety.
    """
    # Adaptive chunk size based on doc length
    if total_pages > 150:
        chunk_size, overlap = 1200, 100
    elif total_pages > 50:
        chunk_size, overlap = 900, 80
    elif total_pages > 20:
        chunk_size, overlap = 600, 60
    else:
        chunk_size, overlap = 400, 50

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)

    # Hard cap — evenly sampled so whole doc stays covered
    MAX_CHUNKS = 400
    if len(chunks) > MAX_CHUNKS:
        step = max(1, len(chunks) // MAX_CHUNKS)
        chunks = chunks[::step][:MAX_CHUNKS]

    total_chunks = len(chunks)
    if progress_cb:
        progress_cb(50, f"Indexing {total_chunks} chunks (0/{total_chunks})...")

    emb = get_embeddings_model()

    # ── True incremental embedding: build FAISS batch by batch ──
    BATCH = 50          # small batches → progress updates feel responsive
    vdb = None
    for i in range(0, total_chunks, BATCH):
        batch = chunks[i : i + BATCH]
        if vdb is None:
            vdb = FAISS.from_texts(batch, emb)
        else:
            vdb.add_texts(batch)
        done = min(i + BATCH, total_chunks)
        if progress_cb:
            pct = 50 + int(45 * done / total_chunks)
            progress_cb(pct, f"Indexing chunks ({done}/{total_chunks})...")

    return vdb, total_chunks


# ==================== AUTH SCREEN ====================
if not st.session_state.authenticated and not st.session_state.is_guest:

    col_anim, col_form = st.columns([1, 1], gap="large")

    with col_anim:
        type_text_logo()
        lottie_data = load_lottie("https://assets10.lottiefiles.com/packages/lf20_jcikwtux.json")
        if lottie_data:
            st_lottie(lottie_data, height=360, key="login_anim")
        else:
            st.markdown(f'<div style="text-align:center;padding-top:60px">{logo_img_tag(120)}</div>', unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:#888;font-size:0.85rem;margin-top:8px;'>Analyze PDFs & Images using AI</div>", unsafe_allow_html=True)

    with col_form:
        st.markdown("<div style='padding-top:55px'></div>", unsafe_allow_html=True)
        tab_login, tab_signup, tab_guest = st.tabs(["🔐 Login", "📝 Sign Up", "👤 Guest"])

        with tab_login:
            st.markdown("#### Welcome back!")
            le = st.text_input("Email", key="login_email", placeholder="you@example.com")
            lp = st.text_input("Password", type="password", key="login_password", placeholder="••••••••")
            cb, cf = st.columns(2)
            with cb:
                if st.button("Login", use_container_width=True, type="primary"):
                    if not le or not lp:
                        st.warning("Please fill in all fields.")
                    else:
                        with st.spinner("Logging in..."):
                            user = login(le, lp)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.is_guest = False
                            st.session_state.user_id = user.uid
                            st.session_state.email = le
                            st.rerun()
            with cf:
                if st.button("Forgot Password?", use_container_width=True):
                    if not le:
                        st.warning("Enter your email above first.")
                    else:
                        with st.spinner("Sending..."):
                            sent = reset_password(le)
                        if sent:
                            st.success("📧 Reset email sent!")

        with tab_signup:
            st.markdown("#### Create your account")
            ne = st.text_input("Email", key="signup_email", placeholder="you@example.com")
            np_ = st.text_input("Password", type="password", key="signup_password", placeholder="Min 6 characters")
            cp = st.text_input("Confirm Password", type="password", key="confirm_password", placeholder="Re-enter password")
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not ne or not np_ or not cp:
                    st.warning("Please fill in all fields.")
                elif len(np_) < 6:
                    st.warning("Password must be at least 6 characters.")
                elif np_ != cp:
                    st.error("❌ Passwords do not match.")
                else:
                    with st.spinner("Creating account..."):
                        user = signup(ne, np_)
                    if user:
                        st.success("✅ Account created! Go to the Login tab.")

        with tab_guest:
            st.markdown("#### Continue without an account")
            st.info("**Guest mode:** Full PDF & Image AI — no account needed.\n\n❌ Chats not saved, clears on refresh.")
            if st.button("👤 Continue as Guest", use_container_width=True, type="primary"):
                st.session_state.is_guest = True
                st.session_state.user_id = "guest"
                st.session_state.email = "Guest"
                st.session_state.guest_messages = []
                st.rerun()

    # ---- Animated Credits Footer ----
    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)
    components.html("""<!DOCTYPE html>
<html>
<head>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
  body {
    margin: 0;
    background: transparent;
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 90px;
    font-family: 'Inter', sans-serif;
  }
  .credits-wrap {
    text-align: center;
    max-width: 700px;
    padding: 0 16px;
  }
  .credits-text {
    font-size: 13px;
    color: #888;
    line-height: 2;
    letter-spacing: 0.3px;
  }
  .word {
    display: inline-block;
    opacity: 0;
    transform: translateY(8px);
    animation: wordIn 0.4s ease forwards;
    margin-right: 4px;
  }
  .highlight {
    font-weight: 600;
    background: linear-gradient(135deg, #6C63FF 0%, #48CAE4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  @keyframes wordIn {
    to { opacity: 1; transform: translateY(0); }
  }
</style>
</head>
<body>
<div class="credits-wrap">
  <div class="credits-text" id="credits"></div>
</div>
<script>
  const parts = [
    { text: "The", highlight: false },
    { text: "Project", highlight: false },
    { text: "Developed", highlight: false },
    { text: "By", highlight: false },
    { text: "Katteboina Mahesh,", highlight: true },
    { text: "Jonalagadda Navaneeth Chowdary", highlight: true },
    { text: "and", highlight: false },
    { text: "Muppalla Chaitra Sahasra", highlight: true },
    { text: "Under", highlight: false },
    { text: "CSE (Data Science)", highlight: true },
    { text: "Department", highlight: false },
    { text: "Guided", highlight: false },
    { text: "by", highlight: false },
    { text: "Rallabandi CH S N P Sairam", highlight: true },
  ];

  const container = document.getElementById('credits');
  parts.forEach((part, i) => {
    const span = document.createElement('span');
    span.className = 'word' + (part.highlight ? ' highlight' : '');
    span.textContent = part.text + ' ';
    span.style.animationDelay = (i * 0.12) + 's';
    container.appendChild(span);
  });
</script>
</body>
</html>""", height=90)

    st.stop()


# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown(f"""
        <div class="sidebar-logo">
            {logo_img_tag(36)}
            <span class="sidebar-logo-text">SLIDESENSE</span>
        </div>""", unsafe_allow_html=True)

    if st.session_state.is_guest:
        st.warning("👤 Guest Mode")
        st.caption("Chats are not saved.")
    else:
        st.success(f"👤 {st.session_state.email}")

    if st.button("🚪 Logout", use_container_width=True):
        for k in defaults:
            st.session_state[k] = defaults[k]
        st.rerun()

    st.divider()
    mode = st.radio("Mode", ["📘 PDF Analyzer", "🖼 Image Q&A"])
    st.session_state.mode = "PDF" if "PDF" in mode else "IMAGE"
    st.divider()

    if st.session_state.is_guest:
        st.markdown("### 💬 Session Chat")
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.guest_messages = []
            st.session_state.vector_db = None
            st.rerun()
        if not st.session_state.current_chat_id:
            st.session_state.current_chat_id = "guest_session"
    else:
        st.markdown("### 💬 Your Chats")
        user_chats = load_user_chats(st.session_state.user_id, st.session_state.mode)
        for chat_id, title in user_chats:
            c1, c2 = st.columns([4, 1])
            if c1.button(title, key=f"open_{chat_id}"):
                st.session_state.current_chat_id = chat_id
                st.session_state.vector_db = None
                st.rerun()
            if c2.button("🗑", key=f"del_{chat_id}"):
                db.collection("users").document(st.session_state.user_id) \
                  .collection("chats").document(chat_id).delete()
                if st.session_state.current_chat_id == chat_id:
                    st.session_state.current_chat_id = None
                st.rerun()
        if st.button("➕ New Chat", use_container_width=True):
            cid = create_new_chat(st.session_state.user_id, st.session_state.mode)
            st.session_state.current_chat_id = cid
            st.session_state.vector_db = None
            st.rerun()


# ==================== WELCOME SCREEN (after login, before chat) ====================
if not st.session_state.current_chat_id:
    st.markdown("""
    <style>
    .ss-wrap {
        display:flex; flex-direction:column;
        align-items:center; justify-content:center;
        padding:60px 0 40px 0;
    }
    .ss-svg { width:220px; height:220px; }

    .slide-back { animation: ssSlideIn 0.8s ease-out forwards; }
    .slide-front { animation: ssSlideIn 0.8s 0.2s ease-out forwards; opacity:0; }

    .ss-node { animation: ssPulse 2s infinite ease-in-out; }
    .ss-node-1 { animation-delay:0.5s; }
    .ss-node-2 { animation-delay:0.8s; }
    .ss-node-3 { animation-delay:1.1s; }

    .ss-arrow {
        stroke-dasharray:100; stroke-dashoffset:100;
        animation: ssDraw 1s 1.2s cubic-bezier(0.16,1,0.3,1) forwards;
    }
    .ss-brand {
        font-size:44px; font-weight:900; letter-spacing:5px;
        background:linear-gradient(135deg,#6C63FF 0%,#48CAE4 100%);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        background-clip:text; margin-top:14px;
        opacity:0; animation: ssFadeUp 0.8s 1.5s forwards;
    }
    .ss-tagline {
        font-size:14px; color:#888; letter-spacing:2px;
        text-transform:uppercase; margin-top:6px;
        opacity:0; animation: ssFade 1s 2s forwards;
    }
    .ss-hint {
        margin-top:28px; font-size:14px; color:#6C63FF;
        opacity:0; animation: ssFade 1s 2.5s forwards;
        background:rgba(108,99,255,0.08);
        border:1px solid rgba(108,99,255,0.3);
        border-radius:10px; padding:10px 24px;
    }

    @keyframes ssSlideIn {
        from { transform:translateX(-30px); opacity:0; }
        to   { transform:translateX(0);     opacity:1; }
    }
    @keyframes ssPulse {
        0%,100% { fill:#0ea5e9; }
        50%      { fill:#38bdf8; filter:drop-shadow(0 0 3px #38bdf8); }
    }
    @keyframes ssDraw { to { stroke-dashoffset:0; } }
    @keyframes ssFadeUp {
        from { opacity:0; transform:translateY(12px); }
        to   { opacity:1; transform:translateY(0); }
    }
    @keyframes ssFade { to { opacity:1; } }
    </style>

    <div class="ss-wrap">
        <svg class="ss-svg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <path class="slide-back"
                d="M30 20 H65 A5 5 0 0 1 70 25 V75 A5 5 0 0 1 65 80 H30 A5 5 0 0 1 25 75 V25 A5 5 0 0 1 30 20"
                fill="#0369a1"/>
            <path class="slide-front"
                d="M35 25 H70 A5 5 0 0 1 75 30 V80 A5 5 0 0 1 70 85 H35 A5 5 0 0 1 30 80 V30 A5 5 0 0 1 35 25"
                fill="#0ea5e9"/>
            <circle cx="55" cy="55" r="22" fill="white" stroke="#0c4a6e" stroke-width="3"/>
            <circle class="ss-node ss-node-1" cx="48" cy="52" r="2" fill="#0ea5e9"/>
            <circle class="ss-node ss-node-2" cx="55" cy="48" r="2" fill="#0ea5e9"/>
            <circle class="ss-node ss-node-3" cx="53" cy="58" r="2" fill="#0ea5e9"/>
            <line x1="48" y1="52" x2="55" y2="48" stroke="#cbd5e1" stroke-width="0.5"/>
            <line x1="55" y1="48" x2="53" y2="58" stroke="#cbd5e1" stroke-width="0.5"/>
            <path class="ss-arrow"
                d="M45 65 L75 35 M75 35 L68 35 M75 35 L75 42"
                stroke="#0ea5e9" stroke-width="4"
                stroke-linecap="round" stroke-linejoin="round" fill="none"/>
        </svg>
        <div class="ss-brand">SLIDESENSE</div>
        <div class="ss-tagline">PDF &amp; Image Q&amp;A</div>
        <div class="ss-hint">👈 Select ➕ New Chat from the sidebar to begin</div>
    </div>
    """, unsafe_allow_html=True)


# ==================== CHAT SCREEN ====================
else:
    img_file = None
    camera_file = None

    if st.session_state.mode == "PDF":
        ac, tc = st.columns([1, 4])
        with ac:
            components.html("""<!DOCTYPE html>
<html><head><style>
body {
  display:flex; flex-direction:column;
  align-items:center; justify-content:center;
  background:transparent; margin:0; padding:8px 0;
  font-family:'Segoe UI',sans-serif;
}
.pdf-card {
  position:relative; width:90px; height:120px;
  background:rgba(255,255,255,0.05);
  backdrop-filter:blur(10px);
  -webkit-backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,0.1);
  border-radius:10px;
  box-shadow:0 8px 32px rgba(0,0,0,0.37);
  overflow:hidden;
  display:flex; flex-direction:column;
  padding:14px 10px; box-sizing:border-box;
}
.skeleton-line {
  height:6px; background:#334155;
  border-radius:4px; margin-bottom:9px;
  position:relative; overflow:hidden; width:100%;
}
.skeleton-line.short { width:55%; }
.skeleton-line::after {
  content:""; position:absolute;
  top:0; left:-100%; width:100%; height:100%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,0.22),transparent);
  animation:shimmer 1.5s infinite;
}
.scanner-beam {
  position:absolute; top:-50px; left:0;
  width:100%; height:40px;
  background:linear-gradient(to bottom,transparent,rgba(56,189,248,0.5));
  border-bottom:2px solid #38bdf8;
  box-shadow:0 5px 15px rgba(56,189,248,0.4);
  animation:scan 2.5s infinite ease-in-out alternate;
  z-index:10;
}
.badge {
  position:absolute; bottom:8px; right:8px;
  background:linear-gradient(135deg,#ef4444,#b91c1c);
  color:white; font-size:8px; letter-spacing:1px;
  font-weight:700; padding:3px 6px;
  border-radius:5px;
  box-shadow:0 4px 10px rgba(239,68,68,0.3);
}
@keyframes scan {
  0%  { top:-50px; opacity:0; }
  10% { opacity:1; }
  90% { opacity:1; }
  100%{ top:100%; opacity:0; }
}
@keyframes shimmer { 100%{ left:100%; } }
</style></head>
<body>
  <div class="pdf-card">
    <div class="scanner-beam"></div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line"></div>
    <div class="skeleton-line short"></div>
    <div class="skeleton-line"></div>
    <div class="badge">PDF</div>
  </div>
</body></html>""", height=140)
        with tc:
            st.markdown("## 📘 PDF Analyzer")
            st.caption("Upload a PDF and ask questions about its content.")
        st.divider()

        pdf = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False)

        # ---- PDF PROCESSING ----
        if pdf and st.session_state.vector_db is None:
            # File size guard — warn if > 50 MB
            pdf_size_mb = len(pdf.getvalue()) / (1024 * 1024)
            if pdf_size_mb > 50:
                st.warning(
                    f"⚠️ This PDF is **{pdf_size_mb:.1f} MB**. "
                    "Very large files may take longer to process."
                )

            anim_slot = st.empty()
            with anim_slot:
                components.html("""<!DOCTYPE html>
<html>
<head>
<style>
  body {
    background: transparent;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 120px;
    margin: 0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
  }
  .loader-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 15px;
  }
  .pdf-box {
    position: relative;
    width: 48px;
    height: 64px;
    background: #1e293b;
    border: 2px solid #38bdf8;
    border-radius: 6px;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
  }
  .laser {
    position: absolute;
    width: 100%;
    height: 3px;
    background: #38bdf8;
    box-shadow: 0 0 12px #38bdf8, 0 0 4px #38bdf8;
    z-index: 2;
    animation: scan-loop 2s ease-in-out infinite;
  }
  .content-lines {
    padding: 10px 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .line {
    height: 3px;
    background: #334155;
    border-radius: 10px;
  }
  .status-text {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 500;
    letter-spacing: 0.5px;
  }
  .dots::after {
    content: '';
    animation: dot-count 1.5s infinite;
  }
  @keyframes scan-loop {
    0%, 100% { top: 5%;  opacity: 0.5; }
    50%       { top: 90%; opacity: 1;   }
  }
  @keyframes dot-count {
    0%   { content: '';    }
    25%  { content: '.';   }
    50%  { content: '..';  }
    75%  { content: '...'; }
    100% { content: '';    }
  }
</style>
</head>
<body>
  <div class="loader-wrapper">
    <div class="pdf-box">
      <div class="laser"></div>
      <div class="content-lines">
        <div class="line" style="width:100%"></div>
        <div class="line" style="width:80%"></div>
        <div class="line" style="width:90%"></div>
        <div class="line" style="width:60%"></div>
        <div class="line" style="width:85%"></div>
      </div>
    </div>
    <div class="status-text">Analyzing the PDF<span class="dots"></span></div>
  </div>
</body>
</html>""", height=120)

            # ── Robust large-PDF processing ──
            progress_bar = st.progress(0, text="📄 Reading PDF pages...")

            try:
                # Step 1 – extract text
                progress_bar.progress(10, text="📄 Extracting text from pages...")
                text, total_pages, warn = extract_pdf_text_safe(pdf)

                if not text.strip():
                    raise ValueError(
                        "No text could be extracted. The PDF may be scanned/image-based "
                        "or password-protected. Try a text-based PDF."
                    )

                if warn:
                    st.warning(f"⚠️ {warn}")

                progress_bar.progress(40, text=f"✅ Read {total_pages} pages. Starting index build...")

                # Step 2 – truly incremental embedding — progress updates after every 50 chunks
                def _progress_cb(pct, msg):
                    progress_bar.progress(min(int(pct), 99), text=msg)

                vdb, num_chunks = build_vector_db(text, total_pages, progress_cb=_progress_cb)
                st.session_state.vector_db = vdb

                progress_bar.progress(100, text="✅ Indexing complete!")
                anim_slot.empty()
                progress_bar.empty()
                st.success(
                    f"✅ Ready! **{total_pages} pages** · **{num_chunks} chunks** indexed. "
                    "Ask your questions below 👇"
                )

            except Exception as _proc_err:
                anim_slot.empty()
                progress_bar.empty()
                st.session_state.vector_db = None
                st.error(f"❌ Failed to process PDF: {_proc_err}")

    else:
        ac, tc = st.columns([1, 4])
        with ac:
            components.html("""<!DOCTYPE html>
<html><head><style>
* { box-sizing:border-box; margin:0; padding:0; }
body {
  display:flex; align-items:center; justify-content:center;
  background:transparent;
  font-family:'Segoe UI', sans-serif;
  height:130px; overflow:hidden;
}
.scene {
  position:relative; width:200px; height:120px;
}

/* Image card */
.img-card {
  position:absolute; left:0; top:10px;
  width:65px; height:90px;
  border-radius:10px;
  background:linear-gradient(160deg,#60a5fa 0%,#3b82f6 40%,#1e40af 100%);
  box-shadow:0 6px 20px rgba(59,130,246,0.4);
  overflow:hidden;
  animation:cardFloat 3s ease-in-out infinite;
}
/* mountain svg inside card */
.img-card svg { width:100%; height:100%; }

/* Search bar */
.search-bar {
  position:absolute; left:38px; top:4px;
  width:150px; height:28px;
  background:white;
  border-radius:20px;
  box-shadow:0 4px 15px rgba(0,0,0,0.15);
  display:flex; align-items:center;
  padding:0 8px; gap:5px;
  animation:slideInDown 0.6s ease-out forwards;
  opacity:0;
}
.search-icon { color:#9ca3af; font-size:11px; flex-shrink:0; }
.search-text {
  font-size:9px; color:#374151; font-weight:500;
  white-space:nowrap; overflow:hidden;
  border-right:1.5px solid #374151;
  width:0;
  animation:typeText 1s 0.8s steps(12) forwards;
}
.search-btn {
  margin-left:auto; width:20px; height:20px;
  background:#3b82f6; border-radius:6px;
  display:flex; align-items:center; justify-content:center;
  font-size:9px; color:white; flex-shrink:0;
  animation:btnPop 0.3s 1.8s ease-out both;
  transform:scale(0);
}

/* Response bubble */
.response-bubble {
  position:absolute; right:0; bottom:0;
  width:100px; height:48px;
  background:linear-gradient(135deg,#34d399,#10b981);
  border-radius:10px 10px 10px 2px;
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 6px 20px rgba(16,185,129,0.35);
  animation:bubblePop 0.5s 2s cubic-bezier(0.175,0.885,0.32,1.275) both;
  transform:scale(0); transform-origin:bottom left;
}
.response-bubble::after {
  content:"";
  position:absolute; bottom:-8px; left:12px;
  width:0; height:0;
  border-left:8px solid transparent;
  border-right:0px solid transparent;
  border-top:8px solid #10b981;
}
.response-inner {
  background:rgba(255,255,255,0.25);
  border-radius:6px; padding:5px 10px;
  font-size:9px; font-weight:600; color:white;
  letter-spacing:0.3px;
  animation:fadeIn 0.3s 2.4s ease-out both; opacity:0;
}

@keyframes cardFloat {
  0%,100% { transform:translateY(0px); }
  50%      { transform:translateY(-5px); }
}
@keyframes slideInDown {
  from { opacity:0; transform:translateY(-10px); }
  to   { opacity:1; transform:translateY(0); }
}
@keyframes typeText {
  from { width:0; }
  to   { width:72px; border-right:none; }
}
@keyframes btnPop {
  to { transform:scale(1); }
}
@keyframes bubblePop {
  to { transform:scale(1); }
}
@keyframes fadeIn {
  to { opacity:1; }
}
</style></head>
<body>
<div class="scene">
  <!-- Image Card with mountain SVG -->
  <div class="img-card">
    <svg viewBox="0 0 65 90" xmlns="http://www.w3.org/2000/svg">
      <rect width="65" height="90" fill="#bfdbfe"/>
      <polygon points="10,80 32,30 54,80" fill="#1e3a8a" opacity="0.9"/>
      <polygon points="0,80 18,45 36,80" fill="#1e40af" opacity="0.8"/>
      <polygon points="28,80 45,42 65,80" fill="#1d4ed8" opacity="0.7"/>
      <polygon points="24,42 32,28 40,42" fill="white" opacity="0.9"/>
      <rect y="60" width="65" height="30" fill="#1e3a8a" opacity="0.5"/>
      <!-- clouds -->
      <ellipse cx="12" cy="18" rx="8" ry="4" fill="white" opacity="0.7"/>
      <ellipse cx="50" cy="14" rx="6" ry="3" fill="white" opacity="0.6"/>
    </svg>
  </div>

  <!-- Search Bar -->
  <div class="search-bar">
    <span class="search-icon">🔍</span>
    <span class="search-text">What is this?</span>
    <div class="search-btn">🔍</div>
  </div>

  <!-- Response Bubble -->
  <div class="response-bubble">
    <div class="response-inner">A Mountain</div>
  </div>
</div>
</body></html>""", height=140)
        with tc:
            st.markdown("## 🖼 Image Q&A")
            st.caption("Upload an image or use your live camera.")
        st.divider()

        img_source = st.radio("Image Source", ["📁 Upload Image", "📷 Instant analyzer"],
                              horizontal=True, key="img_source")

        if img_source == "📁 Upload Image":
            img_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
            if img_file:
                st.image(Image.open(img_file).convert("RGB"), use_container_width=True)
        else:
            st.info("📷 Point your camera and click **'Take Photo'** to capture.")
            camera_file = st.camera_input("Take a photo")
            if camera_file:
                st.image(Image.open(camera_file).convert("RGB"), use_container_width=True)

    # ---- Messages ----
    messages = st.session_state.guest_messages if st.session_state.is_guest else \
               load_messages(st.session_state.user_id, st.session_state.current_chat_id)

    st.markdown("### 💬 Conversation")
    for role, content in messages:
        with st.chat_message("user" if role == "user" else "assistant"):
            if role == "assistant":
                # ── Re-split stored answer so accuracy renders inside the copy block ──
                import re as _re
                _lines = content.strip().splitlines()
                _acc_line = ""
                _clean = content
                for _i, _l in enumerate(_lines):
                    if _l.strip().lower().startswith("accuracy:"):
                        _acc_line = _l.strip()
                        _clean = "\n".join(_lines[:_i] + _lines[_i+1:]).strip()
                        break
                _acc_color = ""
                if _acc_line:
                    _match = _re.search(r'(\d+)', _acc_line)
                    _pct = int(_match.group(1)) if _match else 0
                    _acc_color = "#22c55e" if _pct >= 75 else ("#f59e0b" if _pct >= 45 else "#ef4444")
                render_answer_with_copy(_clean, _acc_line, _acc_color)
            else:
                st.markdown(content)

    question = st.chat_input("Ask something...")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        if st.session_state.is_guest:
            st.session_state.guest_messages.append(("user", question))
        else:
            existing_msgs = load_messages(st.session_state.user_id, st.session_state.current_chat_id)
            if len(existing_msgs) == 0:
                update_chat_title(st.session_state.user_id, st.session_state.current_chat_id, question)
            save_message(st.session_state.user_id, st.session_state.current_chat_id, "user", question)

        if st.session_state.mode == "PDF":
            if st.session_state.vector_db is None:
                answer = "⚠️ Please upload a PDF first."
            else:
                # ── Cache check: skip API call if already answered ──
                _cache_key = f"{st.session_state.current_chat_id}::{question.strip().lower()}"
                if _cache_key in st.session_state.response_cache:
                    answer = st.session_state.response_cache[_cache_key]
                else:
                    with st.spinner("Searching document and generating answer..."):
                        # Retrieve top-3 most relevant chunks; trim each to 400 chars
                        # so total context stays well within token limits
                        raw_docs = st.session_state.vector_db.similarity_search(question, k=3)
                        trimmed_context = "\n\n---\n\n".join(
                            d.page_content[:400] for d in raw_docs
                        )

                        _prompt = ChatPromptTemplate.from_template(
                            "You are a precise AI assistant. Answer ONLY from the context below.\n\n"
                            "Context (excerpts from the uploaded document):\n{context}\n\n"
                            "Question: {input}\n\n"
                            "Instructions:\n"
                            "- Give a clear, concise answer based strictly on the context.\n"
                            "- If the answer is not in the context, say: Information not found in document.\n"
                            "- After your answer, on a NEW line write exactly: Accuracy: XX% "
                            "(your confidence 0–100 based on how well context matches the question).\n"
                            "- Do NOT add any explanation after the Accuracy line."
                        )

                        def _run_pdf_chain():
                            from langchain_core.documents import Document as _Doc
                            _doc = _Doc(page_content=trimmed_context)
                            chain = create_stuff_documents_chain(load_llm(), _prompt)
                            result = chain.invoke({"context": [_doc], "input": question})
                            return result if isinstance(result, str) else result.get("output_text", str(result))

                        answer, err = call_llm_with_retry(_run_pdf_chain)
                        if err:
                            answer = err
                        else:
                            st.session_state.response_cache[_cache_key] = answer
        else:
            active_image = img_file or camera_file
            if not active_image:
                answer = "⚠️ Please upload an image or take a photo first."
            else:
                img_anim_slot = st.empty()
                with img_anim_slot:
                    components.html("""<!DOCTYPE html>
<html>
<head>
<style>
  body {
    background: transparent;
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100px;
    margin: 0;
    font-family: 'Inter', system-ui, sans-serif;
  }
  .qa-box {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }
  .viewport {
    position: relative;
    width: 56px;
    height: 56px;
    background: rgba(99, 102, 241, 0.05);
    border-radius: 4px;
    display: flex;
    justify-content: center;
    align-items: center;
  }
  .viewport::before, .viewport::after {
    content: '';
    position: absolute;
    width: 100%;
    height: 100%;
    border: 2px solid transparent;
    box-sizing: border-box;
    animation: focus-pulse 2s ease-in-out infinite;
  }
  .viewport::before {
    border-top-color: #818cf8;
    border-left-color: #818cf8;
    top: 0; left: 0;
  }
  .viewport::after {
    border-bottom-color: #818cf8;
    border-right-color: #818cf8;
    bottom: 0; right: 0;
  }
  .crosshair {
    width: 16px;
    height: 16px;
    border: 1px solid #38bdf8;
    border-radius: 50%;
    position: absolute;
    display: flex;
    justify-content: center;
    align-items: center;
    animation: ai-search 4s infinite linear;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.4);
  }
  .crosshair::after {
    content: '';
    width: 2px;
    height: 2px;
    background: #38bdf8;
    border-radius: 50%;
  }
  .status {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.5px;
  }
  @keyframes focus-pulse {
    0%, 100% { width: 100%; height: 100%; opacity: 0.8; }
    50%       { width: 85%;  height: 85%;  opacity: 0.3; }
  }
  @keyframes ai-search {
    0%   { transform: translate(-15px, -15px); }
    25%  { transform: translate(15px, -10px);  }
    50%  { transform: translate(10px,  15px);  }
    75%  { transform: translate(-12px, 10px);  }
    100% { transform: translate(-15px, -15px); }
  }
  .dots::after {
    content: '';
    animation: dots 1.5s steps(4, end) infinite;
  }
  @keyframes dots {
    0%,  20% { content: '';    }
    40%      { content: '.';   }
    60%      { content: '..';  }
    80%      { content: '...'; }
  }
</style>
</head>
<body>
  <div class="qa-box">
    <div class="viewport">
      <div class="crosshair"></div>
    </div>
    <div class="status">Analyzing image<span class="dots"></span></div>
  </div>
</body>
</html>""", height=100)

                image_bytes = active_image.getvalue()
                encoded = base64.b64encode(image_bytes).decode("utf-8")
                mime = "image/jpeg" if (camera_file and not img_file) else \
                       ("image/png" if img_file.name.lower().endswith(".png") else "image/jpeg")
                def _run_image_call():
                    resp = load_llm().invoke([HumanMessage(content=[
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                    ])])
                    return resp.content
                answer, err = call_llm_with_retry(_run_image_call)
                if err:
                    answer = err
                img_anim_slot.empty()

        # ── Strip accuracy line from answer for separate rendering ──
        import re
        accuracy_line = ""
        clean_answer = answer
        if st.session_state.mode == "PDF":
            lines = answer.strip().splitlines()
            for i, line in enumerate(lines):
                if line.strip().lower().startswith("accuracy:"):
                    accuracy_line = line.strip()
                    clean_answer = "\n".join(lines[:i] + lines[i+1:]).strip()
                    break

        # Compute accuracy color for new answer
        acc_color = ""
        if accuracy_line:
            match = re.search(r'(\d+)', accuracy_line)
            pct = int(match.group(1)) if match else 0
            acc_color = "#22c55e" if pct >= 75 else ("#f59e0b" if pct >= 45 else "#ef4444")

        with st.chat_message("assistant"):
            render_answer_with_copy(clean_answer, accuracy_line, acc_color)

        # Save the full answer (with accuracy) to storage
        if st.session_state.is_guest:
            st.session_state.guest_messages.append(("assistant", answer))
        else:
            save_message(st.session_state.user_id, st.session_state.current_chat_id, "assistant", answer)

        st.rerun()
