import os
import re
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st

from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
TOP_K = 5

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Current Groq model
GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODERN UI CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       GLOBAL
       ====================================================== */

    .stApp {
        background: #f7f8fc;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3 {
        color: #172033;
    }

    p {
        color: #5f6b7a;
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero {
        background: white;
        padding: 30px 34px;
        border-radius: 18px;
        border: 1px solid #e7eaf0;
        margin-bottom: 24px;
        box-shadow: 0 4px 18px rgba(20, 30, 50, 0.04);
    }

    .hero-title {
        font-size: 32px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 15px;
        color: #6b7280;
        margin: 0;
        line-height: 1.6;
    }


    /* ======================================================
       CARDS
       ====================================================== */

    .card {
        background: white;
        border: 1px solid #e7eaf0;
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 3px 15px rgba(20, 30, 50, 0.035);
    }

    .card-title {
        font-size: 18px;
        font-weight: 650;
        color: #172033;
        margin-bottom: 5px;
    }

    .card-description {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 16px;
        line-height: 1.5;
    }


    /* ======================================================
       METRIC CARDS
       ====================================================== */

    .metric-card {
        background: white;
        border: 1px solid #e7eaf0;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 3px 12px rgba(20, 30, 50, 0.03);
    }

    .metric-number {
        font-size: 25px;
        font-weight: 700;
        color: #172033;
    }

    .metric-label {
        font-size: 13px;
        color: #7b8492;
        margin-top: 4px;
    }


    /* ======================================================
       DOCUMENT SOURCE CARDS
       ====================================================== */

    .source-card {
        background: #fafbfc;
        border: 1px solid #e4e7ec;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
    }

    .source-file {
        font-size: 14px;
        font-weight: 650;
        color: #273142;
    }

    .source-page {
        font-size: 12px;
        color: #7b8492;
        margin-top: 4px;
    }

    .source-text {
        font-size: 13px;
        line-height: 1.65;
        color: #525c6b;
        margin-top: 12px;
        white-space: pre-wrap;
    }


    /* ======================================================
       SOURCE SECTION
       ====================================================== */

    .source-header {
        font-size: 18px;
        font-weight: 650;
        color: #172033;
        margin-top: 28px;
        margin-bottom: 6px;
    }


    /* ======================================================
       CHAT
       ====================================================== */

    .chat-user {
        background: #eef3ff;
        border: 1px solid #dce5ff;
        padding: 16px 18px;
        border-radius: 14px;
        margin: 12px 0;
    }

    .chat-user-label {
        font-size: 11px;
        font-weight: 700;
        color: #5269a6;
        margin-bottom: 6px;
        letter-spacing: 0.5px;
    }

    .chat-user-text {
        font-size: 15px;
        color: #25304a;
        line-height: 1.6;
    }


    .chat-assistant {
        background: white;
        border: 1px solid #e5e8ee;
        padding: 20px;
        border-radius: 14px;
        margin: 12px 0 22px 0;
        box-shadow: 0 3px 12px rgba(20, 30, 50, 0.03);
    }

    .chat-assistant-label {
        font-size: 11px;
        font-weight: 700;
        color: #596579;
        margin-bottom: 8px;
        letter-spacing: 0.5px;
    }

    .chat-assistant-text {
        font-size: 15px;
        line-height: 1.7;
        color: #252b36;
        white-space: pre-wrap;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] {
        background: white;
        border-right: 1px solid #e7eaf0;
    }

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    .sidebar-title {
        font-size: 21px;
        font-weight: 700;
        color: #172033;
        margin-bottom: 5px;
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: #7b8492;
        margin-bottom: 20px;
        line-height: 1.5;
    }


    /* ======================================================
       FILE UPLOADER
       ====================================================== */

    [data-testid="stFileUploader"] {
        background: #fafbfc;
        border: 1px dashed #cbd2dc;
        border-radius: 14px;
        padding: 8px;
    }


    /* ======================================================
       BUTTONS
       ====================================================== */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #d9dee7;
        min-height: 42px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: #8b98ad;
        transform: translateY(-1px);
    }


    /* ======================================================
       INPUTS
       ====================================================== */

    .stTextInput input {
        border-radius: 10px;
        border: 1px solid #d9dee7;
        padding: 12px;
    }

    .stTextInput input:focus {
        border-color: #8b98ad;
        box-shadow: none;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    .streamlit-expanderHeader {
        border-radius: 10px;
        font-weight: 600;
    }


    /* ======================================================
       DIVIDER
       ====================================================== */

    hr {
        border: none;
        border-top: 1px solid #e7eaf0;
        margin: 25px 0;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">
            📚 AI Document Assistant
        </div>

        <p class="hero-subtitle">
            Ask questions about your documents and get answers
            grounded in your uploaded knowledge base.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None

if "processed" not in st.session_state:
    st.session_state.processed = False


# ============================================================
# LOAD EMBEDDING MODEL ONCE
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf(file_path, filename):
    """
    Extract text from PDF.

    Page number is preserved.
    """

    documents = []

    reader = PdfReader(file_path)

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text() or ""

        if text.strip():

            documents.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": page_number
                }
            )

    return documents


def extract_docx(file_path, filename):
    """
    Extract text from DOCX.

    DOCX page numbers are not reliably available,
    so page is stored as None.
    """

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    full_text = "\n".join(paragraphs)

    if not full_text.strip():
        return []

    return [
        {
            "text": full_text,
            "filename": filename,
            "page": None
        }
    ]


def extract_txt(file_path, filename):
    """
    Extract text from TXT.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        text = file.read()

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None
        }
    ]


def extract_md(file_path, filename):
    """
    Extract text from Markdown.
    """

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        text = file.read()

    if not text.strip():
        return []

    return [
        {
            "text": text,
            "filename": filename,
            "page": None
        }
    ]


def extract_document(file_path, filename):
    """
    Select extraction function based on extension.
    """

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(
            file_path,
            filename
        )

    elif extension == ".docx":
        return extract_docx(
            file_path,
            filename
        )

    elif extension == ".txt":
        return extract_txt(
            file_path,
            filename
        )

    elif extension == ".md":
        return extract_md(
            file_path,
            filename
        )

    return []


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(documents):
    """
    Split extracted documents into overlapping chunks.

    Filename and page metadata are preserved.
    """

    chunks = []

    for document in documents:

        text = document["text"]
        filename = document["filename"]
        page = document["page"]

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk = text[start:end].strip()

            if chunk:

                chunks.append(
                    {
                        "text": chunk,
                        "filename": filename,
                        "page": page
                    }
                )

            start += (
                CHUNK_SIZE -
                CHUNK_OVERLAP
            )

    return chunks


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):
    """
    Create embeddings once for all chunks.
    """

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype("float32")


# ============================================================
# CREATE FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):
    """
    Create FAISS cosine-similarity index.

    Normalized embeddings + inner product
    provide cosine similarity.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# KEYWORD SEARCH
# ============================================================

def tokenize(text):
    """
    Simple keyword tokenizer.
    """

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )

    stop_words = {
        "the",
        "is",
        "are",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "for",
        "on",
        "with",
        "what",
        "how",
        "when",
        "where",
        "which",
        "who"
    }

    return [
        word
        for word in words
        if word not in stop_words
        and len(word) > 2
    ]


def keyword_score(
    question,
    chunk_text
):
    """
    Score chunk based on matching
    important question words.
    """

    question_words = set(
        tokenize(question)
    )

    chunk_words = set(
        tokenize(chunk_text)
    )

    if not question_words:
        return 0.0

    matches = question_words.intersection(
        chunk_words
    )

    return (
        len(matches) /
        len(question_words)
    )


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    question,
    top_k=TOP_K
):
    """
    Combine semantic FAISS search
    and keyword search.
    """

    chunks = st.session_state.chunks

    index = st.session_state.faiss_index

    if not chunks or index is None:
        return []

    # --------------------------------------------------------
    # SEMANTIC SEARCH
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores, semantic_indices = (
        index.search(
            question_embedding,
            min(
                top_k * 2,
                len(chunks)
            )
        )
    )

    semantic_scores = semantic_scores[0]
    semantic_indices = semantic_indices[0]

    semantic_results = {}

    for score, index_number in zip(
        semantic_scores,
        semantic_indices
    ):

        semantic_results[
            int(index_number)
        ] = float(score)

    # --------------------------------------------------------
    # KEYWORD SEARCH
    # --------------------------------------------------------

    results = []

    for index_number, chunk in enumerate(
        chunks
    ):

        semantic_score = semantic_results.get(
            index_number,
            0.0
        )

        keyword = keyword_score(
            question,
            chunk["text"]
        )

        # Hybrid weighting
        hybrid_score = (
            0.75 * semantic_score
            +
            0.25 * keyword
        )

        results.append(
            {
                "text": chunk["text"],
                "filename": chunk["filename"],
                "page": chunk["page"],
                "semantic_score": semantic_score,
                "keyword_score": keyword,
                "hybrid_score": hybrid_score
            }
        )

    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():
    """
    Read GROQ_API_KEY from Streamlit secrets.
    """

    api_key = st.secrets.get(
        "GROQ_API_KEY"
    )

    if not api_key:
        return None

    return Groq(
        api_key=api_key
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    retrieved_chunks
):
    """
    Generate answer using only
    retrieved document context.
    """

    client = get_groq_client()

    if client is None:

        return (
            "GROQ_API_KEY is not configured. "
            "Please add GROQ_API_KEY to "
            "Streamlit secrets."
        )

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        page_text = ""

        if chunk["page"] is not None:

            page_text = (
                f", Page {chunk['page']}"
            )

        context_parts.append(
            f"""
SOURCE {i}
Filename: {chunk['filename']}{page_text}

Content:
{chunk['text']}
"""
        )

    context = "\n".join(
        context_parts
    )

    system_prompt = """
You are an AI document assistant.

Answer the user's question ONLY using
the provided document context.

Rules:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the context,
   clearly say:

   "The information is not available in
   the provided documents."

4. Keep the answer clear and concise.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],

        temperature=0,
        max_tokens=700
    )

    return response.choices[0].message.content


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents(documents):
    """
    Complete processing pipeline:

    extraction
        ↓
    chunking
        ↓
    embeddings
        ↓
    FAISS
    """

    chunks = chunk_text(
        documents
    )

    if not chunks:
        return False

    with st.spinner(
        "Creating document embeddings..."
    ):

        embeddings = create_embeddings(
            chunks
        )

        index = create_faiss_index(
            embeddings
        )

    st.session_state.documents = documents

    st.session_state.chunks = chunks

    st.session_state.embeddings = embeddings

    st.session_state.faiss_index = index

    st.session_state.processed = True

    return True


# ============================================================
# GOOGLE DRIVE
# ============================================================

def load_from_google_drive(url):
    """
    Download public Google Drive file or folder.

    Supports:
    PDF
    DOCX
    TXT
    MD
    """

    temp_directory = tempfile.mkdtemp()

    try:

        # ----------------------------------------------------
        # GOOGLE DRIVE FOLDER
        # ----------------------------------------------------

        if "/folders/" in url:

            downloaded = gdown.download_folder(
                url=url,
                output=temp_directory,
                quiet=True
            )

            if not downloaded:
                return []

            file_paths = []

            for path in downloaded:

                path = Path(path)

                if path.suffix.lower() in {
                    ".pdf",
                    ".docx",
                    ".txt",
                    ".md"
                }:

                    file_paths.append(path)

        # ----------------------------------------------------
        # GOOGLE DRIVE FILE
        # ----------------------------------------------------

        else:

            downloaded_path = gdown.download(
                url=url,
                output=temp_directory,
                quiet=True
            )

            if not downloaded_path:
                return []

            downloaded_path = Path(
                downloaded_path
            )

            if downloaded_path.is_dir():

                file_paths = [
                    path
                    for path in downloaded_path.rglob("*")
                    if path.suffix.lower() in {
                        ".pdf",
                        ".docx",
                        ".txt",
                        ".md"
                    }
                ]

            else:

                file_paths = [
                    downloaded_path
                ]

        # ----------------------------------------------------
        # EXTRACT DOCUMENTS
        # ----------------------------------------------------

        documents = []

        for file_path in file_paths:

            if file_path.suffix.lower() not in {
                ".pdf",
                ".docx",
                ".txt",
                ".md"
            }:
                continue

            filename = file_path.name

            extracted = extract_document(
                str(file_path),
                filename
            )

            documents.extend(
                extracted
            )

        return documents

    except Exception as e:

        st.error(
            f"Could not load Google Drive content: {e}"
        )

        return []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div class="sidebar-title">
        Document Sources
    </div>

    <div class="sidebar-subtitle">
        Upload files or connect a public
        Google Drive source.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOCAL UPLOAD
# ============================================================

st.sidebar.markdown(
    "### 📁 Local Documents"
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=[
        "pdf",
        "docx",
        "txt",
        "md"
    ],
    accept_multiple_files=True,
    label_visibility="collapsed"
)


# ============================================================
# GOOGLE DRIVE INPUT
# ============================================================

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### ☁️ Google Drive"
)

drive_url = st.sidebar.text_input(
    "Google Drive URL",
    placeholder="Paste a public Drive file or folder link"
)

st.sidebar.caption(
    "Supported: PDF, DOCX, TXT and MD"
)


# ============================================================
# PROCESS BUTTON
# ============================================================

process_button = st.sidebar.button(
    "⚡ Process Documents",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PROCESS BUTTON LOGIC
# ============================================================

if process_button:

    all_documents = []

    # --------------------------------------------------------
    # LOCAL UPLOADS
    # --------------------------------------------------------

    if uploaded_files:

        for uploaded_file in uploaded_files:

            file_suffix = Path(
                uploaded_file.name
            ).suffix.lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=file_suffix
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name

            extracted = extract_document(
                temp_path,
                uploaded_file.name
            )

            all_documents.extend(
                extracted
            )

    # --------------------------------------------------------
    # GOOGLE DRIVE
    # --------------------------------------------------------

    if drive_url.strip():

        with st.spinner(
            "Loading documents from Google Drive..."
        ):

            drive_documents = (
                load_from_google_drive(
                    drive_url.strip()
                )
            )

            all_documents.extend(
                drive_documents
            )

    # --------------------------------------------------------
    # PROCESS EVERYTHING
    # --------------------------------------------------------

    if all_documents:

        success = process_documents(
            all_documents
        )

        if success:

            st.sidebar.success(
                "Documents processed successfully."
            )

    else:

        st.sidebar.warning(
            "No supported documents found."
        )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.documents:

    st.markdown(
        """
        <div class="card">

            <div class="card-title">
                📄 Knowledge Base
            </div>

            <div class="card-description">
                Documents currently available
                to the assistant.
            </div>

        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # UNIQUE DOCUMENTS
    # --------------------------------------------------------

    unique_documents = {}

    for document in (
        st.session_state.documents
    ):

        filename = document["filename"]

        if filename not in unique_documents:

            unique_documents[filename] = {
                "pages": set(),
                "characters": 0
            }

        page = document["page"]

        if page is not None:

            unique_documents[
                filename
            ]["pages"].add(page)

        unique_documents[
            filename
        ]["characters"] += len(
            document["text"]
        )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    total_documents = len(
        unique_documents
    )

    total_chunks = len(
        st.session_state.chunks
    )

    total_characters = sum(
        item["characters"]
        for item in unique_documents.values()
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {total_documents}
                </div>

                <div class="metric-label">
                    Documents
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {total_chunks}
                </div>

                <div class="metric-label">
                    Chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {total_characters:,}
                </div>

                <div class="metric-label">
                    Characters
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # DOCUMENT LIST
    # --------------------------------------------------------

    for filename, information in (
        unique_documents.items()
    ):

        pages = information["pages"]

        if pages:

            page_info = (
                f"{len(pages)} page(s)"
            )

        else:

            page_info = (
                "Page information unavailable"
            )

        st.markdown(
            f"""
            <div class="source-card">

                <div class="source-file">
                    📄 {filename}
                </div>

                <div class="source-page">
                    {page_info}
                    &nbsp; • &nbsp;
                    {information["characters"]:,}
                    characters
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# ASK QUESTION SECTION
# ============================================================

st.markdown(
    """
    <div class="card">

        <div class="card-title">
            💬 Ask Your Documents
        </div>

        <div class="card-description">
            Ask a question and the assistant will
            search your documents for the most
            relevant information.
        </div>

    """,
    unsafe_allow_html=True
)


question = st.text_input(
    "Question",
    placeholder=(
        "e.g. What is the annual leave policy?"
    ),
    label_visibility="collapsed"
)


ask_button = st.button(
    "🔍 Ask Assistant",
    type="primary",
    use_container_width=True
)


st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# QUESTION ANSWERING
# ============================================================

if ask_button:

    if not st.session_state.processed:

        st.warning(
            "Please process your documents first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ----------------------------------------------------
        # HYBRID SEARCH
        # ----------------------------------------------------

        with st.spinner(
            "Searching your documents..."
        ):

            retrieved_chunks = hybrid_search(
                question,
                top_k=TOP_K
            )

        if not retrieved_chunks:

            st.warning(
                "No relevant information was found."
            )

        else:

            # ------------------------------------------------
            # GROQ
            # ------------------------------------------------

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            # ------------------------------------------------
            # USER MESSAGE
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="chat-user">

                    <div class="chat-user-label">
                        YOU
                    </div>

                    <div class="chat-user-text">
                        {question}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # ASSISTANT MESSAGE
            # ------------------------------------------------

            st.markdown(
                f"""
                <div class="chat-assistant">

                    <div class="chat-assistant-label">
                        🤖 AI DOCUMENT ASSISTANT
                    </div>

                    <div class="chat-assistant-text">
                        {answer}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            st.markdown(
                """
                <div class="source-header">
                    📚 Retrieved Sources
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                f"{len(retrieved_chunks)} relevant "
                "source chunks retrieved"
            )

            # ------------------------------------------------
            # SOURCE CARDS
            # ------------------------------------------------

            for i, source in enumerate(
                retrieved_chunks,
                start=1
            ):

                if source["page"] is not None:

                    page_text = (
                        f"Page {source['page']}"
                    )

                else:

                    page_text = (
                        "Page information unavailable"
                    )

                with st.expander(
                    f"Source {i}  ·  "
                    f"{source['filename']}  ·  "
                    f"{page_text}"
                ):

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-file">
                                📄 {source['filename']}
                            </div>

                            <div class="source-page">
                                {page_text}
                            </div>

                            <div class="source-text">
                                {source['text']}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.caption(
                        f"Semantic: "
                        f"{source['semantic_score']:.3f}"
                        f"   •   "
                        f"Keyword: "
                        f"{source['keyword_score']:.3f}"
                        f"   •   "
                        f"Hybrid: "
                        f"{source['hybrid_score']:.3f}"
                    )
