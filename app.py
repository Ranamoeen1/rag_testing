```python
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
import requests
import streamlit as st
from docx import Document
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "RAG Document Q&A Assistant"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

GROQ_MODEL = "openai/gpt-oss-120b"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

SEMANTIC_TOP_K = 8
KEYWORD_TOP_K = 8
FINAL_TOP_K = 6

SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35

GOOGLE_DRIVE_API_BASE = (
    "https://www.googleapis.com/drive/v3"
)

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
    ".markdown",
}

SUPPORTED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "text/plain": ".txt",
    "text/markdown": ".md",
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODERN SAAS UI STYLING
# ============================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 5%,
                rgba(99, 102, 241, 0.07),
                transparent 28%
            ),
            radial-gradient(
                circle at 10% 20%,
                rgba(14, 165, 233, 0.05),
                transparent 25%
            ),
            #f8fafc;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin-bottom: 2rem;
    }

    .sidebar-logo {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #111827;
        color: white;
        font-size: 19px;
        font-weight: 700;
    }

    .sidebar-brand-text {
        font-size: 15px;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }

    .sidebar-brand-subtitle {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 2px;
    }

    .sidebar-section {
        font-size: 11px;
        font-weight: 700;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1.5rem 0 0.7rem 0;
    }

    .config-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px;
        margin-bottom: 12px;
    }

    .config-label {
        font-size: 11px;
        color: #94a3b8;
        margin-bottom: 3px;
    }

    .config-value {
        font-size: 12px;
        color: #334155;
        font-weight: 600;
        word-break: break-word;
    }

    /* ========================================================
       HERO
       ======================================================== */

    .hero {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 22px;
        padding: 34px 38px;
        margin-bottom: 22px;
        box-shadow:
            0 1px 2px rgba(15, 23, 42, 0.03),
            0 8px 30px rgba(15, 23, 42, 0.03);
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        padding: 6px 11px;
        border-radius: 999px;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #475569;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 15px;
    }

    .hero-title {
        font-size: 32px;
        line-height: 1.15;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.035em;
        margin-bottom: 9px;
    }

    .hero-description {
        color: #64748b;
        font-size: 14px;
        line-height: 1.7;
        max-width: 720px;
    }

    /* ========================================================
       SECTION HEADERS
       ======================================================== */

    .section-title {
        font-size: 18px;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.015em;
        margin-bottom: 4px;
    }

    .section-description {
        font-size: 12px;
        color: #94a3b8;
        margin-bottom: 13px;
    }

    /* ========================================================
       SOURCE CARDS
       ======================================================== */

    .upload-card,
    .drive-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 15px;
        box-shadow:
            0 1px 2px rgba(15, 23, 42, 0.02),
            0 6px 20px rgba(15, 23, 42, 0.025);
    }

    .source-option-title {
        font-size: 13px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 4px;
    }

    .source-option-description {
        font-size: 11px;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 13px;
    }

    div[data-testid="stFileUploader"] {
        width: 100%;
    }

    div[data-testid="stFileUploader"] section {
        border: 1.5px dashed #cbd5e1 !important;
        border-radius: 16px !important;
        background: #f8fafc !important;
        padding: 25px 20px !important;
        transition: all 0.2s ease;
    }

    div[data-testid="stFileUploader"] section:hover {
        border-color: #94a3b8 !important;
        background: #f1f5f9 !important;
    }

    div[data-testid="stFileUploader"] small {
        color: #94a3b8 !important;
    }

    /* ========================================================
       INPUTS
       ======================================================== */

    div[data-baseweb="input"] {
        border-radius: 11px !important;
    }

    div[data-baseweb="input"] > div {
        border-radius: 11px !important;
        border-color: #cbd5e1 !important;
        background: #ffffff !important;
    }

    div[data-baseweb="input"] input {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        border-radius: 11px !important;
        min-height: 42px !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        transition:
            transform 0.15s ease,
            box-shadow 0.15s ease,
            background 0.15s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
    }

    button[kind="primary"] {
        background: #111827 !important;
        border: 1px solid #111827 !important;
        color: #ffffff !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12);
    }

    button[kind="primary"]:hover {
        background: #1f2937 !important;
        border-color: #1f2937 !important;
        box-shadow: 0 7px 18px rgba(15, 23, 42, 0.16);
    }

    /* ========================================================
       ALERTS
       ======================================================== */

    div[data-testid="stAlert"] {
        border-radius: 12px !important;
        border-width: 1px !important;
        font-size: 13px !important;
    }

    div[data-testid="stStatusWidget"] {
        border-radius: 14px !important;
    }

    /* ========================================================
       METRIC CARDS
       ======================================================== */

    .metric-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 17px 18px;
        height: 100%;
        box-shadow: 0 3px 15px rgba(15, 23, 42, 0.025);
    }

    .metric-icon {
        font-size: 18px;
        margin-bottom: 8px;
    }

    .metric-label {
        font-size: 11px;
        font-weight: 600;
        color: #94a3b8;
        margin-bottom: 3px;
    }

    .metric-value {
        font-size: 21px;
        font-weight: 700;
        color: #0f172a;
    }

    /* ========================================================
       CHAT
       ======================================================== */

    .chat-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 25px;
        margin-bottom: 13px;
    }

    .chat-status {
        font-size: 11px;
        font-weight: 600;
        color: #64748b;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 999px;
        padding: 6px 10px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 17px;
        margin-bottom: 12px;
        padding: 7px 0;
    }

    div[data-testid="stChatMessage"]
    [data-testid="stMarkdownContainer"] {
        font-size: 13.5px;
        line-height: 1.75;
    }

    div[data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        padding: 14px 16px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.025);
    }

    div[data-testid="stChatInput"] {
        margin-top: 15px;
    }

    div[data-testid="stChatInput"] > div {
        border-radius: 15px !important;
        border: 1px solid #cbd5e1 !important;
        background: #ffffff !important;
        box-shadow:
            0 4px 20px rgba(15, 23, 42, 0.05) !important;
    }

    div[data-testid="stChatInput"] textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 13px !important;
    }

    /* ========================================================
       RETRIEVED SOURCES
       ======================================================== */

    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 14px !important;
        background: #f8fafc !important;
        margin-top: 12px;
    }

    div[data-testid="stExpander"] summary {
        font-size: 12px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }

    .source-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 13px;
        padding: 14px;
        margin: 8px 0;
    }

    .source-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 7px;
    }

    .source-number {
        width: 25px;
        height: 25px;
        border-radius: 8px;
        background: #f1f5f9;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #475569;
        font-size: 11px;
        font-weight: 700;
    }

    .source-name {
        font-size: 12px;
        font-weight: 650;
        color: #334155;
    }

    .source-meta {
        font-size: 10px;
        color: #94a3b8;
        margin-bottom: 9px;
    }

    .source-text {
        background: #f8fafc;
        border: 1px solid #f1f5f9;
        border-radius: 9px;
        padding: 11px;
        color: #475569;
        font-size: 11px;
        line-height: 1.65;
        white-space: pre-wrap;
    }

    /* ========================================================
       EMPTY STATE
       ======================================================== */

    .empty-state {
        background: #ffffff;
        border: 1px dashed #cbd5e1;
        border-radius: 18px;
        padding: 42px 25px;
        text-align: center;
        margin-top: 15px;
    }

    .empty-icon {
        font-size: 30px;
        margin-bottom: 10px;
    }

    .empty-title {
        font-size: 14px;
        font-weight: 650;
        color: #334155;
        margin-bottom: 5px;
    }

    .empty-description {
        font-size: 12px;
        color: #94a3b8;
    }

    /* ========================================================
       FOOTER
       ======================================================== */

    .footer {
        text-align: center;
        padding-top: 35px;
        color: #94a3b8;
        font-size: 10px;
    }

    @media (max-width: 768px) {

        .main .block-container {
            padding: 1rem 0.8rem 3rem 0.8rem;
        }

        .hero {
            padding: 25px 22px;
            border-radius: 18px;
        }

        .hero-title {
            font-size: 25px;
        }

        .hero-description {
            font-size: 13px;
        }

        .upload-card,
        .drive-card {
            padding: 12px;
        }

        div[data-testid="stFileUploader"] section {
            padding: 18px 10px !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():

    defaults = {
        "documents": [],
        "chunks": [],
        "embeddings": None,
        "tfidf_vectorizer": None,
        "tfidf_matrix": None,
        "processed": False,
        "chat_history": [],
        "file_signature": None,

        # Google Drive state
        "drive_files": [],
        "drive_signature": None,
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key() -> str:

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if api_key:
        return api_key.strip()

    try:

        if "GROQ_API_KEY" in st.secrets:

            return str(
                st.secrets[
                    "GROQ_API_KEY"
                ]
            ).strip()

    except Exception:
        pass

    return ""


# ============================================================
# GOOGLE DRIVE API KEY
# ============================================================

def get_google_drive_api_key() -> str:
    """
    Get GOOGLE_DRIVE_API_KEY from environment variables
    or Streamlit secrets.

    This key is only used for publicly accessible
    Google Drive files/folders.
    """

    api_key = os.getenv(
        "GOOGLE_DRIVE_API_KEY"
    )

    if api_key:
        return api_key.strip()

    try:

        if "GOOGLE_DRIVE_API_KEY" in st.secrets:

            return str(
                st.secrets[
                    "GOOGLE_DRIVE_API_KEY"
                ]
            ).strip()

    except Exception:
        pass

    return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(
    text: str
) -> str:

    if not text:
        return ""

    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(
    file_bytes: bytes,
    filename: str
) -> List[Dict[str, Any]]:

    pages = []

    try:

        reader = PdfReader(
            BytesIO(file_bytes)
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            try:

                text = (
                    page.extract_text()
                    or ""
                )

                text = clean_text(
                    text
                )

                if text:

                    pages.append(
                        {
                            "text": text,
                            "metadata": {
                                "filename": filename,
                                "file_type": "pdf",
                                "page": page_number,
                                "source": "google_drive"
                                if filename.startswith(
                                    "gdrive:"
                                )
                                else "upload",
                            },
                        }
                    )

            except Exception as page_error:

                st.warning(
                    f"Could not extract page "
                    f"{page_number} from "
                    f"{filename}: {page_error}"
                )

    except Exception as error:

        raise ValueError(
            f"Failed to read PDF "
            f"'{filename}': {error}"
        )

    return pages


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx_text(
    file_bytes: bytes,
    filename: str
) -> List[Dict[str, Any]]:

    try:

        document = Document(
            BytesIO(file_bytes)
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        text = "\n\n".join(
            paragraphs
        )

        text = clean_text(
            text
        )

        if not text:
            return []

        return [
            {
                "text": text,
                "metadata": {
                    "filename": filename,
                    "file_type": "docx",
                    "page": None,
                    "source": "google_drive"
                    if filename.startswith(
                        "gdrive:"
                    )
                    else "upload",
                },
            }
        ]

    except Exception as error:

        raise ValueError(
            f"Failed to read DOCX "
            f"'{filename}': {error}"
        )


# ============================================================
# TXT / MARKDOWN EXTRACTION
# ============================================================

def extract_text_file(
    file_bytes: bytes,
    filename: str,
    file_type: str
) -> List[Dict[str, Any]]:

    try:

        try:
            text = file_bytes.decode(
                "utf-8"
            )

        except UnicodeDecodeError:

            text = file_bytes.decode(
                "latin-1"
            )

        text = clean_text(
            text
        )

        if not text:
            return []

        return [
            {
                "text": text,
                "metadata": {
                    "filename": filename,
                    "file_type": file_type,
                    "page": None,
                    "source": "google_drive"
                    if filename.startswith(
                        "gdrive:"
                    )
                    else "upload",
                },
            }
        ]

    except Exception as error:

        raise ValueError(
            f"Failed to read "
            f"'{filename}': {error}"
        )


# ============================================================
# GENERIC DOCUMENT EXTRACTION
# ============================================================

def extract_document(
    file
) -> List[Dict[str, Any]]:

    filename = file.name

    extension = os.path.splitext(
        filename
    )[1].lower()

    file_bytes = file.getvalue()

    if extension == ".pdf":

        return extract_pdf_text(
            file_bytes,
            filename
        )

    elif extension == ".docx":

        return extract_docx_text(
            file_bytes,
            filename
        )

    elif extension == ".txt":

        return extract_text_file(
            file_bytes,
            filename,
            "txt"
        )

    elif extension in [
        ".md",
        ".markdown"
    ]:

        return extract_text_file(
            file_bytes,
            filename,
            "markdown"
        )

    else:

        raise ValueError(
            f"Unsupported file type: "
            f"{extension or 'unknown'}"
        )


# ============================================================
# CHUNKING
# ============================================================

def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> List[str]:

    if not text:
        return []

    words = text.split()

    words_per_chunk = max(
        50,
        chunk_size // 5
    )

    overlap_words = max(
        10,
        overlap // 5
    )

    if overlap_words >= words_per_chunk:

        overlap_words = (
            words_per_chunk // 3
        )

    chunks = []

    start = 0

    while start < len(words):

        end = min(
            start + words_per_chunk,
            len(words)
        )

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(
                chunk
            )

        if end >= len(words):
            break

        start = (
            end - overlap_words
        )

    return chunks


def chunk_documents(
    extracted_documents:
    List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    all_chunks = []

    for document in extracted_documents:

        text = document["text"]

        metadata = document[
            "metadata"
        ]

        text_chunks = (
            split_text_into_chunks(
                text
            )
        )

        for chunk_index, chunk in enumerate(
            text_chunks
        ):

            chunk_metadata = (
                metadata.copy()
            )

            chunk_metadata[
                "chunk_id"
            ] = chunk_index

            all_chunks.append(
                {
                    "text": chunk,
                    "metadata": chunk_metadata,
                }
            )

    return all_chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks: List[Dict[str, Any]]
) -> np.ndarray:

    if not chunks:

        raise ValueError(
            "No chunks available "
            "for embedding."
        )

    model = load_embedding_model()

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.astype(
        "float32"
    )


# ============================================================
# SEARCH INDEX
# ============================================================

def build_search_index(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray
) -> Tuple[
    TfidfVectorizer,
    Any
]:

    if not chunks:

        raise ValueError(
            "Cannot build an index "
            "without chunks."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )

    tfidf_matrix = (
        vectorizer.fit_transform(
            texts
        )
    )

    return (
        vectorizer,
        tfidf_matrix
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    top_k: int = SEMANTIC_TOP_K
) -> List[Dict[str, Any]]:

    if (
        not chunks
        or embeddings is None
    ):
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype(
        "float32"
    )

    scores = (
        embeddings
        @ query_embedding
    )

    top_k = min(
        top_k,
        len(chunks)
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "index": int(index),
                "score": float(
                    scores[index]
                ),
                "text": chunks[index][
                    "text"
                ],
                "metadata": chunks[index][
                    "metadata"
                ],
                "search_type": "semantic",
            }
        )

    return results


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(
    query: str,
    chunks: List[Dict[str, Any]],
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_k: int = KEYWORD_TOP_K
) -> List[Dict[str, Any]]:

    if (
        not chunks
        or vectorizer is None
        or tfidf_matrix is None
    ):
        return []

    query_vector = (
        vectorizer.transform(
            [query]
        )
    )

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix
    ).flatten()

    top_k = min(
        top_k,
        len(chunks)
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        if scores[index] <= 0:
            continue

        results.append(
            {
                "index": int(index),
                "score": float(
                    scores[index]
                ),
                "text": chunks[index][
                    "text"
                ],
                "metadata": chunks[index][
                    "metadata"
                ],
                "search_type": "keyword",
            }
        )

    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def normalize_scores(
    results: List[Dict[str, Any]]
) -> Dict[int, float]:

    if not results:
        return {}

    scores = [
        item["score"]
        for item in results
    ]

    minimum = min(scores)

    maximum = max(scores)

    if maximum == minimum:

        return {
            item["index"]: 1.0
            for item in results
        }

    return {
        item["index"]:
        (
            item["score"]
            - minimum
        )
        /
        (
            maximum
            - minimum
        )
        for item in results
    }


def hybrid_search(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_k: int = FINAL_TOP_K
) -> List[Dict[str, Any]]:

    semantic_results = (
        semantic_search(
            query=query,
            chunks=chunks,
            embeddings=embeddings,
            top_k=SEMANTIC_TOP_K,
        )
    )

    keyword_results = (
        keyword_search(
            query=query,
            chunks=chunks,
            vectorizer=vectorizer,
            tfidf_matrix=tfidf_matrix,
            top_k=KEYWORD_TOP_K,
        )
    )

    semantic_scores = (
        normalize_scores(
            semantic_results
        )
    )

    keyword_scores = (
        normalize_scores(
            keyword_results
        )
    )

    candidate_indices = set(
        semantic_scores.keys()
    ).union(
        keyword_scores.keys()
    )

    combined_results = []

    for index in candidate_indices:

        semantic_score = (
            semantic_scores.get(
                index,
                0.0
            )
        )

        keyword_score = (
            keyword_scores.get(
                index,
                0.0
            )
        )

        hybrid_score = (
            SEMANTIC_WEIGHT
            * semantic_score
            +
            KEYWORD_WEIGHT
            * keyword_score
        )

        combined_results.append(
            {
                "index": index,
                "semantic_score":
                    semantic_score,
                "keyword_score":
                    keyword_score,
                "hybrid_score":
                    hybrid_score,
                "text":
                    chunks[index][
                        "text"
                    ],
                "metadata":
                    chunks[index][
                        "metadata"
                    ],
            }
        )

    combined_results.sort(
        key=lambda x:
            x["hybrid_score"],
        reverse=True,
    )

    return combined_results[:top_k]


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(
    retrieved_chunks:
    List[Dict[str, Any]]
) -> str:

    context_parts = []

    for number, result in enumerate(
        retrieved_chunks,
        start=1
    ):

        metadata = result[
            "metadata"
        ]

        filename = metadata.get(
            "filename",
            "Unknown file"
        )

        page = metadata.get(
            "page"
        )

        if page:

            source = (
                f"{filename}, "
                f"page {page}"
            )

        else:

            source = filename

        context_parts.append(
            f"""
--- SOURCE {number} ---
File: {source}

Content:
{result["text"]}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def generate_answer(
    question: str,
    retrieved_chunks:
    List[Dict[str, Any]]
) -> str:

    api_key = get_groq_api_key()

    if not api_key:

        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets "
            "or your environment."
        )

    if not retrieved_chunks:

        return (
            "The answer was not found "
            "in the uploaded documents."
        )

    client = Groq(
        api_key=api_key
    )

    context = build_context(
        retrieved_chunks
    )

    system_prompt = """
You are a document question-answering assistant.

Your ONLY source of truth is the DOCUMENT CONTEXT provided
by the application.

Strict rules:

1. Answer ONLY using information found in the provided context.
2. Do NOT use outside knowledge.
3. Do NOT invent facts, numbers, policies, dates, names,
   or explanations that are not supported by the context.
4. If the answer cannot be found in the provided context,
   respond exactly with:

"The answer was not found in the uploaded documents."

5. If the context contains only part of the answer, clearly
   state what is supported and what is not available.
6. Keep answers clear and useful.
7. When appropriate, mention the source filename and page number.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer the user's question using ONLY the document context.
"""

    completion = (
        client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0.0,
            max_completion_tokens=1000,
        )
    )

    answer = (
        completion
        .choices[0]
        .message
        .content
    )

    if not answer:

        raise ValueError(
            "Groq returned an empty response."
        )

    return answer.strip()


# ============================================================
# LOCAL FILE SIGNATURE
# ============================================================

def get_file_signature(
    files
) -> Tuple:

    return tuple(
        (
            file.name,
            file.size,
            getattr(
                file,
                "type",
                None
            ),
        )
        for file in files
    )


# ============================================================
# GOOGLE DRIVE HELPERS
# ============================================================

def extract_drive_id(
    url: str
) -> Tuple[str, str]:

    """
    Extract a Google Drive file/folder ID from common
    Google Drive URL formats.

    Returns:
        ("folder", folder_id)
        ("file", file_id)
    """

    url = url.strip()

    if not url:
        raise ValueError(
            "Please enter a Google Drive link."
        )

    # Folder:
    # https://drive.google.com/drive/folders/FOLDER_ID
    folder_match = re.search(
        r"/folders/([a-zA-Z0-9_-]+)",
        url
    )

    if folder_match:

        return (
            "folder",
            folder_match.group(1)
        )

    # File:
    # https://drive.google.com/file/d/FILE_ID/view
    file_match = re.search(
        r"/file/d/([a-zA-Z0-9_-]+)",
        url
    )

    if file_match:

        return (
            "file",
            file_match.group(1)
        )

    # Open:
    # https://drive.google.com/open?id=FILE_ID
    query_match = re.search(
        r"[?&]id=([a-zA-Z0-9_-]+)",
        url
    )

    if query_match:

        return (
            "file",
            query_match.group(1)
        )

    # Direct Drive URL containing /d/
    direct_match = re.search(
        r"/d/([a-zA-Z0-9_-]+)",
        url
    )

    if direct_match:

        return (
            "file",
            direct_match.group(1)
        )

    raise ValueError(
        "Invalid Google Drive link. "
        "Please provide a Drive file or folder URL."
    )


def drive_api_request(
    endpoint: str,
    params: Dict[str, Any],
    timeout: int = 30
) -> requests.Response:

    api_key = get_google_drive_api_key()

    if not api_key:

        raise ValueError(
            "Google Drive is not configured. "
            "Add GOOGLE_DRIVE_API_KEY to your "
            "Streamlit Secrets."
        )

    params = dict(params)

    params["key"] = api_key

    url = (
        f"{GOOGLE_DRIVE_API_BASE}"
        f"/{endpoint}"
    )

    try:

        response = requests.get(
            url,
            params=params,
            timeout=timeout,
        )

    except requests.RequestException as error:

        raise ConnectionError(
            f"Could not connect to Google Drive: "
            f"{error}"
        )

    if response.status_code in [
        401,
        403
    ]:

        raise PermissionError(
            "Google Drive denied access to this "
            "file or folder. Make sure it is shared "
            "as 'Anyone with the link' and that the "
            "Google Drive API is enabled."
        )

    if response.status_code == 404:

        raise FileNotFoundError(
            "The Google Drive file or folder "
            "could not be found."
        )

    if not response.ok:

        try:

            error_data = (
                response.json()
            )

            message = (
                error_data
                .get("error", {})
                .get(
                    "message",
                    response.text
                )
            )

        except Exception:

            message = response.text

        raise RuntimeError(
            f"Google Drive API error: "
            f"{message}"
        )

    return response


def get_drive_file_metadata(
    file_id: str
) -> Dict[str, Any]:

    response = drive_api_request(
        f"files/{file_id}",
        {
            "fields": (
                "id,name,mimeType,size,"
                "modifiedTime,webViewLink"
            )
        }
    )

    return response.json()


def list_drive_folder_files(
    folder_id: str
) -> List[Dict[str, Any]]:

    """
    List files directly inside a public Drive folder.

    Pagination is handled so larger folders are supported.
    """

    files = []

    page_token = None

    while True:

        params = {
            "q": (
                f"'{folder_id}' in parents "
                f"and trashed = false"
            ),
            "pageSize": 1000,
            "fields": (
                "nextPageToken,"
                "files(id,name,mimeType,size,"
                "modifiedTime,webViewLink)"
            ),
            "orderBy": "name",
        }

        if page_token:

            params[
                "pageToken"
            ] = page_token

        response = drive_api_request(
            "files",
            params
        )

        data = response.json()

        files.extend(
            data.get(
                "files",
                []
            )
        )

        page_token = data.get(
            "nextPageToken"
        )

        if not page_token:
            break

    return files


def is_supported_drive_file(
    file_metadata: Dict[str, Any]
) -> bool:

    name = file_metadata.get(
        "name",
        ""
    )

    extension = os.path.splitext(
        name
    )[1].lower()

    mime_type = file_metadata.get(
        "mimeType",
        ""
    )

    return (
        extension in SUPPORTED_EXTENSIONS
        or
        mime_type in SUPPORTED_MIME_TYPES
    )


def get_drive_extension(
    file_metadata: Dict[str, Any]
) -> str:

    name = file_metadata.get(
        "name",
        ""
    )

    extension = os.path.splitext(
        name
    )[1].lower()

    if extension in SUPPORTED_EXTENSIONS:

        return extension

    mime_type = file_metadata.get(
        "mimeType",
        ""
    )

    return SUPPORTED_MIME_TYPES.get(
        mime_type,
        ""
    )


def download_drive_file(
    file_metadata: Dict[str, Any]
) -> bytes:

    file_id = file_metadata[
        "id"
    ]

    response = drive_api_request(
        f"files/{file_id}",
        {
            "alt": "media"
        },
        timeout=60,
    )

    return response.content


def create_drive_document_records(
    file_metadata: Dict[str, Any],
    file_bytes: bytes
) -> List[Dict[str, Any]]:

    """
    Convert a downloaded Google Drive file into the same
    document record structure used by local uploads.
    """

    original_name = file_metadata.get(
        "name",
        "unknown"
    )

    extension = get_drive_extension(
        file_metadata
    )

    if not extension:

        raise ValueError(
            f"Unsupported Drive file: "
            f"{original_name}"
        )

    # Prefix is internal only and allows the extraction
    # functions to mark the source as Google Drive.
    safe_filename = (
        f"gdrive:{original_name}"
    )

    if extension == ".pdf":

        return extract_pdf_text(
            file_bytes,
            safe_filename
        )

    if extension == ".docx":

        return extract_docx_text(
            file_bytes,
            safe_filename
        )

    if extension == ".txt":

        return extract_text_file(
            file_bytes,
            safe_filename,
            "txt"
        )

    if extension in [
        ".md",
        ".markdown"
    ]:

        return extract_text_file(
            file_bytes,
            safe_filename,
            "markdown"
        )

    raise ValueError(
        f"Unsupported Drive file: "
        f"{original_name}"
    )


def load_google_drive_link(
    drive_url: str
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    int
]:

    """
    Load supported documents from a public Google Drive
    file or folder.

    Returns:

        extracted_documents
        drive_file_metadata
        unsupported_count
    """

    resource_type, resource_id = (
        extract_drive_id(
            drive_url
        )
    )

    # --------------------------------------------------------
    # Determine whether the link points to a file or folder.
    # --------------------------------------------------------

    if resource_type == "folder":

        drive_files = (
            list_drive_folder_files(
                resource_id
            )
        )

    else:

        metadata = (
            get_drive_file_metadata(
                resource_id
            )
        )

        drive_files = [
            metadata
        ]

    if not drive_files:

        raise ValueError(
            "No files were found at the "
            "provided Google Drive link."
        )

    supported_files = []

    unsupported_count = 0

    for file_metadata in drive_files:

        # Skip folders inside a folder.
        if (
            file_metadata.get(
                "mimeType"
            )
            == "application/vnd.google-apps.folder"
        ):

            continue

        if is_supported_drive_file(
            file_metadata
        ):

            supported_files.append(
                file_metadata
            )

        else:

            unsupported_count += 1

    if not supported_files:

        raise ValueError(
            "No supported files were found. "
            "Google Drive folders must contain "
            "PDF, DOCX, TXT or MD files."
        )

    extracted_documents = []

    successfully_loaded = []

    for file_metadata in supported_files:

        filename = file_metadata.get(
            "name",
            "Unknown file"
        )

        try:

            file_bytes = (
                download_drive_file(
                    file_metadata
                )
            )

            if not file_bytes:

                st.warning(
                    f"'{filename}' is empty."
                )

                continue

            documents = (
                create_drive_document_records(
                    file_metadata,
                    file_bytes
                )
            )

            if not documents:

                st.warning(
                    f"No readable text found "
                    f"in '{filename}'."
                )

                continue

            extracted_documents.extend(
                documents
            )

            successfully_loaded.append(
                file_metadata
            )

        except Exception as error:

            st.warning(
                f"Could not load "
                f"'{filename}': {error}"
            )

    if not extracted_documents:

        raise ValueError(
            "The Drive files were found, "
            "but no readable text could be extracted."
        )

    return (
        extracted_documents,
        successfully_loaded,
        unsupported_count,
    )


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_application():

    st.session_state.documents = []

    st.session_state.chunks = []

    st.session_state.embeddings = None

    st.session_state.tfidf_vectorizer = None

    st.session_state.tfidf_matrix = None

    st.session_state.processed = False

    st.session_state.chat_history = []

    st.session_state.file_signature = None

    st.session_state.drive_files = []

    st.session_state.drive_signature = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">R</div>
            <div>
                <div class="sidebar-brand-text">
                    RAG Assistant
                </div>
                <div class="sidebar-brand-subtitle">
                    Document Intelligence
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">System</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="config-card">
            <div class="config-label">
                Embedding Model
            </div>
            <div class="config-value">
                {EMBEDDING_MODEL_NAME}
            </div>
        </div>

        <div class="config-card">
            <div class="config-label">
                Language Model
            </div>
            <div class="config-value">
                {GROQ_MODEL}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Retrieval</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="config-card">
            <div class="config-label">
                Semantic Weight
            </div>
            <div class="config-value">
                {SEMANTIC_WEIGHT:.0%}
            </div>
        </div>

        <div class="config-card">
            <div class="config-label">
                Keyword Weight
            </div>
            <div class="config-value">
                {KEYWORD_WEIGHT:.0%}
            </div>
        </div>

        <div class="config-card">
            <div class="config-label">
                Retrieved Chunks
            </div>
            <div class="config-value">
                {FINAL_TOP_K}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-section">Knowledge Base</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.processed:

        st.success(
            "Knowledge base ready"
        )

        st.metric(
            "Indexed chunks",
            len(
                st.session_state.chunks
            ),
        )

    else:

        st.info(
            "Upload documents or load a "
            "Google Drive folder."
        )

    st.markdown(
        '<div style="height:8px"></div>',
        unsafe_allow_html=True,
    )

    if st.button(
        "Reset Knowledge Base",
        use_container_width=True,
    ):

        reset_application()

        st.rerun()


# ============================================================
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <div class="hero-badge">
            ✦ AI-Powered Document Intelligence
        </div>

        <div class="hero-title">
            Ask your documents anything.
        </div>

        <div class="hero-description">
            Upload documents or connect a public Google Drive
            file/folder and get grounded answers using
            semantic search, keyword retrieval and
            Groq-powered generation.
        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DOCUMENT SOURCES
# ============================================================

st.markdown(
    '<div class="section-title">Add Knowledge</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        Choose how you want to add documents to your
        knowledge base.
    </div>
    """,
    unsafe_allow_html=True,
)

source_tab_1, source_tab_2 = st.tabs(
    [
        "📄 Upload Files",
        "☁️ Google Drive",
    ]
)


# ============================================================
# LOCAL FILE UPLOAD
# ============================================================

with source_tab_1:

    st.markdown(
        """
        <div class="upload-card">

            <div class="source-option-title">
                Upload documents
            </div>

            <div class="source-option-description">
                Upload one or more PDF, DOCX, TXT or
                Markdown files directly from your computer.
            </div>

        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "markdown",
        ],
        accept_multiple_files=True,
        help=(
            "Upload multiple PDF, DOCX, TXT "
            "or Markdown documents."
        ),
        label_visibility="collapsed",
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# GOOGLE DRIVE
# ============================================================

with source_tab_2:

    st.markdown(
        """
        <div class="drive-card">

            <div class="source-option-title">
                Connect Google Drive
            </div>

            <div class="source-option-description">
                Paste a public Google Drive file or folder
                link. Supported files are PDF, DOCX, TXT
                and Markdown.
            </div>

        """,
        unsafe_allow_html=True,
    )

    drive_url = st.text_input(
        "Google Drive link",
        placeholder=(
            "https://drive.google.com/drive/folders/..."
        ),
        label_visibility="collapsed",
        key="google_drive_url",
    )

    drive_load_button = st.button(
        "☁️ Load from Drive",
        use_container_width=True,
    )

    st.caption(
        "The Drive item must be shared as "
        "'Anyone with the link'. Private Drive content "
        "requires OAuth authentication and is not accessed "
        "by this deployment-friendly mode."
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# GOOGLE DRIVE LOADING
# ============================================================

if drive_load_button:

    if not drive_url.strip():

        st.warning(
            "Please paste a Google Drive file or "
            "folder link."
        )

    elif not get_google_drive_api_key():

        st.error(
            "Google Drive is not configured. "
            "Add GOOGLE_DRIVE_API_KEY to Streamlit Secrets."
        )

    else:

        with st.status(
            "Connecting to Google Drive...",
            expanded=True,
        ) as drive_status:

            try:

                st.write(
                    "🔗 Validating Drive link..."
                )

                (
                    drive_documents,
                    drive_files,
                    unsupported_count,
                ) = load_google_drive_link(
                    drive_url
                )

                st.write(
                    f"✓ Found {len(drive_files)} "
                    f"supported file(s)."
                )

                if unsupported_count:

                    st.info(
                        f"Skipped {unsupported_count} "
                        f"unsupported file(s)."
                    )

                st.session_state.drive_files = (
                    drive_files
                )

                st.session_state.drive_signature = (
                    drive_url.strip()
                )

                # ------------------------------------------------
                # Create chunks
                # ------------------------------------------------

                st.write(
                    "✂️ Splitting Drive documents..."
                )

                chunks = chunk_documents(
                    drive_documents
                )

                if not chunks:

                    raise ValueError(
                        "No chunks were created "
                        "from the Drive documents."
                    )

                st.write(
                    f"✓ Created {len(chunks)} "
                    f"chunks."
                )

                # ------------------------------------------------
                # Create embeddings
                # ------------------------------------------------

                st.write(
                    "🧠 Creating local embeddings..."
                )

                embeddings = create_embeddings(
                    chunks
                )

                st.write(
                    "✓ Embeddings created."
                )

                # ------------------------------------------------
                # Build search index
                # ------------------------------------------------

                st.write(
                    "🔎 Building hybrid search index..."
                )

                (
                    vectorizer,
                    tfidf_matrix,
                ) = build_search_index(
                    chunks,
                    embeddings,
                )

                # ------------------------------------------------
                # Store in same existing session state
                # ------------------------------------------------

                st.session_state.documents = (
                    drive_documents
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.embeddings = (
                    embeddings
                )

                st.session_state.tfidf_vectorizer = (
                    vectorizer
                )

                st.session_state.tfidf_matrix = (
                    tfidf_matrix
                )

                st.session_state.processed = (
                    True
                )

                st.session_state.chat_history = []

                status_text = (
                    f"Successfully indexed "
                    f"{len(drive_files)} file(s) "
                    f"and {len(chunks)} chunks."
                )

                drive_status.update(
                    label=status_text,
                    state="complete",
                    expanded=False,
                )

                st.success(
                    status_text
                )

                # ------------------------------------------------
                # Show Drive file list
                # ------------------------------------------------

                with st.expander(
                    "☁️ Loaded Google Drive Files",
                    expanded=False,
                ):

                    for file_metadata in (
                        drive_files
                    ):

                        st.markdown(
                            f"**{file_metadata.get('name', 'Unknown')}**"
                        )

                        st.caption(
                            f"MIME type: "
                            f"{file_metadata.get('mimeType', 'Unknown')}"
                        )

            except PermissionError as error:

                drive_status.update(
                    label="Google Drive access denied",
                    state="error",
                    expanded=True,
                )

                st.error(
                    str(error)
                )

            except FileNotFoundError as error:

                drive_status.update(
                    label="Google Drive item not found",
                    state="error",
                    expanded=True,
                )

                st.error(
                    str(error)
                )

            except Exception as error:

                drive_status.update(
                    label="Google Drive loading failed",
                    state="error",
                    expanded=True,
                )

                st.error(
                    f"Could not load Google Drive files: "
                    f"{error}"
                )


# ============================================================
# LOCAL FILE SUMMARY
# ============================================================

if uploaded_files:

    file_count = len(
        uploaded_files
    )

    total_size = sum(
        file.size
        for file in uploaded_files
    )

    total_size_mb = (
        total_size /
        (1024 * 1024)
    )

    col1, col2, col3 = st.columns(
        3
    )

    with col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📄</div>
                <div class="metric-label">
                    DOCUMENTS
                </div>
                <div class="metric-value">
                    {file_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">💾</div>
                <div class="metric-label">
                    TOTAL SIZE
                </div>
                <div class="metric-value">
                    {total_size_mb:.1f} MB
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-icon">⚡</div>
                <div class="metric-label">
                    SOURCE
                </div>
                <div class="metric-value">
                    Local
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# LOCAL UPLOAD STATE
# ============================================================

if uploaded_files:

    current_signature = (
        get_file_signature(
            uploaded_files
        )
    )

    if (
        st.session_state.file_signature
        is not None
        and
        current_signature
        !=
        st.session_state.file_signature
    ):

        st.session_state.processed = False

        st.info(
            "Your local document selection changed. "
            "Process the documents again to update "
            "the knowledge base."
        )


# ============================================================
# PROCESS LOCAL DOCUMENTS
# ============================================================

process_button = st.button(
    "Process Uploaded Documents",
    type="primary",
    use_container_width=True,
)


if process_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one document "
            "before processing."
        )

    else:

        with st.status(
            "Building your knowledge base...",
            expanded=True,
        ) as status:

            try:

                # ------------------------------------------------
                # STEP 1: Extract text
                # ------------------------------------------------

                st.write(
                    "📖 Extracting document text..."
                )

                extracted_documents = []

                for uploaded_file in (
                    uploaded_files
                ):

                    extension = os.path.splitext(
                        uploaded_file.name
                    )[1].lower()

                    if (
                        extension
                        not in
                        SUPPORTED_EXTENSIONS
                    ):

                        st.warning(
                            f"Skipping unsupported file: "
                            f"{uploaded_file.name}"
                        )

                        continue

                    try:

                        documents = (
                            extract_document(
                                uploaded_file
                            )
                        )

                        if not documents:

                            st.warning(
                                f"No text found in "
                                f"'{uploaded_file.name}'."
                            )

                            continue

                        extracted_documents.extend(
                            documents
                        )

                    except Exception as error:

                        st.error(
                            f"Could not process "
                            f"'{uploaded_file.name}': "
                            f"{error}"
                        )

                if not extracted_documents:

                    raise ValueError(
                        "No readable text was extracted "
                        "from the uploaded documents."
                    )

                st.write(
                    f"✓ Extracted text from "
                    f"{len(extracted_documents)} "
                    f"document sections."
                )

                # ------------------------------------------------
                # STEP 2: Chunk
                # ------------------------------------------------

                st.write(
                    "✂️ Splitting documents into chunks..."
                )

                chunks = chunk_documents(
                    extracted_documents
                )

                if not chunks:

                    raise ValueError(
                        "No chunks were created "
                        "from the documents."
                    )

                st.write(
                    f"✓ Created {len(chunks)} "
                    f"text chunks."
                )

                # ------------------------------------------------
                # STEP 3: Embeddings
                # ------------------------------------------------

                st.write(
                    "🧠 Creating local embeddings..."
                )

                embeddings = create_embeddings(
                    chunks
                )

                st.write(
                    f"✓ Created embeddings with "
                    f"{embeddings.shape[1]} dimensions."
                )

                # ------------------------------------------------
                # STEP 4: Search index
                # ------------------------------------------------

                st.write(
                    "🔎 Building hybrid search index..."
                )

                (
                    vectorizer,
                    tfidf_matrix,
                ) = build_search_index(
                    chunks,
                    embeddings,
                )

                st.write(
                    "✓ Semantic and keyword "
                    "indexes ready."
                )

                # ------------------------------------------------
                # STEP 5: Session state
                # ------------------------------------------------

                st.session_state.documents = (
                    extracted_documents
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.embeddings = (
                    embeddings
                )

                st.session_state.tfidf_vectorizer = (
                    vectorizer
                )

                st.session_state.tfidf_matrix = (
                    tfidf_matrix
                )

                st.session_state.processed = (
                    True
                )

                st.session_state.file_signature = (
                    get_file_signature(
                        uploaded_files
                    )
                )

                st.session_state.chat_history = []

                status.update(
                    label="Knowledge base ready",
                    state="complete",
                    expanded=False,
                )

                st.success(
                    f"Successfully indexed "
                    f"{len(chunks)} chunks."
                )

            except Exception as error:

                status.update(
                    label="Processing failed",
                    state="error",
                    expanded=True,
                )

                st.error(
                    f"Processing failed: {error}"
                )


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

if st.session_state.processed:

    st.markdown(
        f"""
        <div style="
            background:#f0fdf4;
            border:1px solid #bbf7d0;
            border-radius:12px;
            padding:11px 14px;
            margin-top:10px;
            color:#166534;
            font-size:12px;
            font-weight:600;
        ">
            ✓ Knowledge base active ·
            {len(st.session_state.chunks)}
            chunks ready for retrieval
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# CHAT SECTION
# ============================================================

st.markdown(
    """
    <div class="chat-header">
        <div>
            <div class="section-title">
                Ask your documents
            </div>
            <div class="section-description"
                 style="margin-bottom:0;">
                Ask questions and inspect exactly which
                document passages were retrieved.
            </div>
        </div>

        <div class="chat-status">
            ● Grounded RAG
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EMPTY CHAT STATE
# ============================================================

if (
    not st.session_state.chat_history
    and
    not st.session_state.processed
):

    st.markdown(
        """
        <div class="empty-state">

            <div class="empty-icon">
                💬
            </div>

            <div class="empty-title">
                Your AI assistant is ready when you are.
            </div>

            <div class="empty-description">
                Upload documents or connect Google Drive,
                process them, and start asking questions.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in (
    st.session_state.chat_history
):

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
            and
            message.get("sources")
        ):

            with st.expander(
                "🔎  Retrieved Sources",
                expanded=False,
            ):

                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):

                    metadata = (
                        source["metadata"]
                    )

                    filename = metadata.get(
                        "filename",
                        "Unknown file",
                    )

                    # Remove the internal gdrive prefix
                    # from displayed source names.
                    display_filename = (
                        filename.replace(
                            "gdrive:",
                            "",
                            1
                        )
                    )

                    page = metadata.get(
                        "page"
                    )

                    if page:

                        location = (
                            f"{display_filename} · "
                            f"Page {page}"
                        )

                    else:

                        location = (
                            display_filename
                        )

                    st.markdown(
                        f"""
                        <div class="source-card">

                            <div class="source-header">

                                <div class="source-number">
                                    {index}
                                </div>

                                <div class="source-name">
                                    {location}
                                </div>

                            </div>

                            <div class="source-meta">
                                Hybrid
                                {source["hybrid_score"]:.3f}
                                &nbsp; · &nbsp;
                                Semantic
                                {source["semantic_score"]:.3f}
                                &nbsp; · &nbsp;
                                Keyword
                                {source["keyword_score"]:.3f}
                            </div>

                            <div class="source-text">
                                {source["text"]}
                            </div>

                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.processed:

        st.warning(
            "Please upload and process documents "
            "or load a Google Drive source before "
            "asking a question."
        )

    else:

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):

            st.markdown(
                question
            )

        # ----------------------------------------------------
        # Assistant response
        # ----------------------------------------------------

        with st.chat_message(
            "assistant"
        ):

            with st.spinner(
                "Searching your knowledge base..."
            ):

                try:

                    # ------------------------------------------------
                    # Hybrid retrieval
                    # ------------------------------------------------

                    retrieved_chunks = (
                        hybrid_search(
                            query=question,
                            chunks=(
                                st.session_state.chunks
                            ),
                            embeddings=(
                                st.session_state.embeddings
                            ),
                            vectorizer=(
                                st.session_state
                                .tfidf_vectorizer
                            ),
                            tfidf_matrix=(
                                st.session_state
                                .tfidf_matrix
                            ),
                            top_k=FINAL_TOP_K,
                        )
                    )

                    if not retrieved_chunks:

                        answer = (
                            "The answer was not found "
                            "in the uploaded documents."
                        )

                    else:

                        # ------------------------------------------------
                        # Existing Groq generation
                        # ------------------------------------------------

                        answer = generate_answer(
                            question=question,
                            retrieved_chunks=(
                                retrieved_chunks
                            ),
                        )

                    # ------------------------------------------------
                    # Answer
                    # ------------------------------------------------

                    st.markdown(
                        answer
                    )

                    # ------------------------------------------------
                    # Sources
                    # ------------------------------------------------

                    if retrieved_chunks:

                        with st.expander(
                            "🔎  Retrieved Sources",
                            expanded=True,
                        ):

                            for index, source in enumerate(
                                retrieved_chunks,
                                start=1,
                            ):

                                metadata = (
                                    source["metadata"]
                                )

                                filename = metadata.get(
                                    "filename",
                                    "Unknown file",
                                )

                                display_filename = (
                                    filename.replace(
                                        "gdrive:",
                                        "",
                                        1
                                    )
                                )

                                page = metadata.get(
                                    "page"
                                )

                                if page:

                                    location = (
                                        f"{display_filename} · "
                                        f"Page {page}"
                                    )

                                else:

                                    location = (
                                        display_filename
                                    )

                                st.markdown(
                                    f"""
                                    <div class="source-card">

                                        <div class="source-header">

                                            <div class="source-number">
                                                {index}
                                            </div>

                                            <div class="source-name">
                                                {location}
                                            </div>

                                        </div>

                                        <div class="source-meta">
                                            Hybrid
                                            {source["hybrid_score"]:.3f}
                                            &nbsp; · &nbsp;
                                            Semantic
                                            {source["semantic_score"]:.3f}
                                            &nbsp; · &nbsp;
                                            Keyword
                                            {source["keyword_score"]:.3f}
                                        </div>

                                        <div class="source-text">
                                            {source["text"]}
                                        </div>

                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # ------------------------------------------------
                    # Save assistant response
                    # ------------------------------------------------

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": (
                                retrieved_chunks
                            ),
                        }
                    )

                except Exception as error:

                    error_message = (
                        f"Sorry, an error occurred: "
                        f"{error}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        RAG Document Q&A · Local Upload + Google Drive ·
        Semantic + Keyword Retrieval · Grounded Generation
    </div>
    """,
    unsafe_allow_html=True,
)
```
