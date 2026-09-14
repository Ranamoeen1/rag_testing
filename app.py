import html
import os
import re
import tempfile
from pathlib import Path

import faiss
import gdown
import numpy as np
import streamlit as st

from docx import Document
from groq import Groq
from groq import APIConnectionError, APIStatusError, AuthenticationError
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "AI Document Assistant"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 5

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Current production Groq model.
# openai/gpt-oss-20b is currently supported by Groq.
GROQ_MODEL = "openai/gpt-oss-20b"


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".md",
}


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
<style>

.stApp {
    background: #f6f8fb;
}

.main .block-container {
    max-width: 1180px;
    padding-top: 32px;
    padding-bottom: 60px;
}

/* ----------------------------------------------------------
   HEADER
   ---------------------------------------------------------- */

.app-header {
    background: #ffffff;
    border: 1px solid #e4e8ef;
    border-radius: 18px;
    padding: 30px 34px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);
}

.app-title {
    color: #172033;
    font-size: 31px;
    font-weight: 750;
    line-height: 1.2;
}

.app-subtitle {
    color: #6b7280;
    font-size: 15px;
    line-height: 1.6;
    margin-top: 9px;
}

/* ----------------------------------------------------------
   SECTIONS
   ---------------------------------------------------------- */

.section {
    background: #ffffff;
    border: 1px solid #e4e8ef;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 22px;
    box-shadow: 0 3px 16px rgba(15, 23, 42, 0.035);
}

.section-title {
    color: #172033;
    font-size: 19px;
    font-weight: 700;
    margin-bottom: 5px;
}

.section-description {
    color: #737d8c;
    font-size: 14px;
    line-height: 1.55;
    margin-bottom: 18px;
}

/* ----------------------------------------------------------
   METRICS
   ---------------------------------------------------------- */

.metric-card {
    background: #f9fafc;
    border: 1px solid #e5e9ef;
    border-radius: 13px;
    padding: 18px 12px;
    text-align: center;
    min-height: 90px;
}

.metric-value {
    color: #172033;
    font-size: 25px;
    font-weight: 750;
}

.metric-label {
    color: #7b8492;
    font-size: 12px;
    margin-top: 4px;
}

/* ----------------------------------------------------------
   DOCUMENTS
   ---------------------------------------------------------- */

.document-card {
    background: #fafbfc;
    border: 1px solid #e4e8ef;
    border-radius: 11px;
    padding: 14px 16px;
    margin-top: 9px;
}

.document-name {
    color: #202938;
    font-size: 14px;
    font-weight: 650;
}

.document-meta {
    color: #7b8492;
    font-size: 12px;
    margin-top: 4px;
}

/* ----------------------------------------------------------
   CHAT
   ---------------------------------------------------------- */

.user-message {
    background: #eef4ff;
    border: 1px solid #dce7ff;
    border-radius: 14px;
    padding: 16px 18px;
    margin: 12px 0;
}

.user-label {
    color: #5570ad;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 0.6px;
    margin-bottom: 7px;
}

.user-text {
    color: #27344f;
    font-size: 15px;
    line-height: 1.6;
}

.ai-message {
    background: #ffffff;
    border: 1px solid #e2e7ee;
    border-radius: 14px;
    padding: 20px;
    margin: 12px 0 24px 0;
    box-shadow: 0 3px 13px rgba(15, 23, 42, 0.035);
}

.ai-label {
    color: #596579;
    font-size: 11px;
    font-weight: 750;
    letter-spacing: 0.6px;
    margin-bottom: 9px;
}

.ai-text {
    color: #252c38;
    font-size: 15px;
    line-height: 1.75;
}

/* ----------------------------------------------------------
   SOURCES
   ---------------------------------------------------------- */

.sources-title {
    color: #172033;
    font-size: 19px;
    font-weight: 700;
    margin-top: 28px;
    margin-bottom: 4px;
}

.sources-description {
    color: #7a8494;
    font-size: 13px;
    margin-bottom: 12px;
}

.source-box {
    background: #fafbfc;
    border: 1px solid #e4e8ef;
    border-radius: 11px;
    padding: 16px;
}

.source-file {
    color: #202938;
    font-size: 14px;
    font-weight: 650;
}

.source-page {
    color: #7b8492;
    font-size: 12px;
    margin-top: 4px;
}

.source-text {
    color: #525d6d;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 12px;
    white-space: pre-wrap;
}

/* ----------------------------------------------------------
   SIDEBAR
   ---------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e4e8ef;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 28px;
}

.sidebar-title {
    color: #172033;
    font-size: 21px;
    font-weight: 750;
    margin-bottom: 5px;
}

.sidebar-description {
    color: #7a8494;
    font-size: 13px;
    line-height: 1.55;
    margin-bottom: 20px;
}

.sidebar-heading {
    color: #303b4d;
    font-size: 14px;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 8px;
}

/* ----------------------------------------------------------
   UPLOADER
   ---------------------------------------------------------- */

[data-testid="stFileUploader"] {
    background: #fafbfc;
    border: 1px dashed #cbd3df;
    border-radius: 12px;
    padding: 7px;
}

/* ----------------------------------------------------------
   TEXT INPUT
   ---------------------------------------------------------- */

.stTextInput input {
    background: #ffffff;
    color: #202938;
    border: 1px solid #d9dfe8;
    border-radius: 10px;
    min-height: 44px;
}

.stTextInput input:focus {
    border-color: #8996a9;
    box-shadow: 0 0 0 1px #8996a9;
}

/* ----------------------------------------------------------
   BUTTONS
   ---------------------------------------------------------- */

.stButton > button {
    border-radius: 10px;
    min-height: 43px;
    font-weight: 650;
    border: 1px solid #d7dde6;
}

.stButton > button:hover {
    border-color: #9aa6b7;
}

/* ----------------------------------------------------------
   EXPANDERS
   ---------------------------------------------------------- */

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e1e6ed;
    border-radius: 11px;
    margin-bottom: 8px;
}

/* ----------------------------------------------------------
   FOOTER / STREAMLIT CHROME
   ---------------------------------------------------------- */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-header">'
    '<div class="app-title">📚 AI Document Assistant</div>'
    '<div class="app-subtitle">'
    'Upload documents, build a searchable knowledge base, '
    'and ask questions using AI-powered document retrieval.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION_STATE = {
    "documents": [],
    "chunks": [],
    "embeddings": None,
    "faiss_index": None,
    "processed": False,
    "processed_signature": None,
}


for key, default_value in DEFAULT_SESSION_STATE.items():

    if key not in st.session_state:

        st.session_state[key] = default_value


# ============================================================
# EMBEDDING MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


try:

    embedding_model = load_embedding_model()

except Exception as error:

    st.error(
        "The embedding model could not be loaded."
    )

    st.caption(
        f"Technical details: {error}"
    )

    st.stop()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(
    file_path,
    filename,
):

    documents = []

    try:

        reader = PdfReader(
            file_path
        )

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = page.extract_text() or ""

            text = text.strip()

            if not text:
                continue

            documents.append(
                {
                    "text": text,
                    "filename": filename,
                    "page": page_number,
                }
            )

    except Exception as error:

        raise ValueError(
            f"Could not extract PDF '{filename}': {error}"
        )

    return documents


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(
    file_path,
    filename,
):

    try:

        document = Document(
            file_path
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:

                paragraphs.append(
                    text
                )

        full_text = "\n".join(
            paragraphs
        )

        if not full_text.strip():
            return []

        return [
            {
                "text": full_text,
                "filename": filename,
                "page": None,
            }
        ]

    except Exception as error:

        raise ValueError(
            f"Could not extract DOCX '{filename}': {error}"
        )


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(
    file_path,
    filename,
):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:

            text = file.read()

        if not text.strip():
            return []

        return [
            {
                "text": text,
                "filename": filename,
                "page": None,
            }
        ]

    except Exception as error:

        raise ValueError(
            f"Could not extract TXT '{filename}': {error}"
        )


# ============================================================
# MARKDOWN EXTRACTION
# ============================================================

def extract_md(
    file_path,
    filename,
):

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as file:

            text = file.read()

        if not text.strip():
            return []

        return [
            {
                "text": text,
                "filename": filename,
                "page": None,
            }
        ]

    except Exception as error:

        raise ValueError(
            f"Could not extract Markdown '{filename}': {error}"
        )


# ============================================================
# DOCUMENT EXTRACTION ROUTER
# ============================================================

def extract_document(
    file_path,
    filename,
):

    extension = Path(
        filename
    ).suffix.lower()

    if extension == ".pdf":

        return extract_pdf(
            file_path,
            filename,
        )

    if extension == ".docx":

        return extract_docx(
            file_path,
            filename,
        )

    if extension == ".txt":

        return extract_txt(
            file_path,
            filename,
        )

    if extension == ".md":

        return extract_md(
            file_path,
            filename,
        )

    return []


# ============================================================
# TEXT CHUNKING
# ============================================================

def chunk_text(
    documents,
):

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    if step <= 0:

        raise ValueError(
            "CHUNK_OVERLAP must be smaller than CHUNK_SIZE."
        )

    for document in documents:

        text = document["text"]

        filename = document["filename"]

        page = document["page"]

        if not text:
            continue

        start = 0

        while start < len(text):

            end = start + CHUNK_SIZE

            chunk = text[start:end].strip()

            if chunk:

                chunks.append(
                    {
                        "text": chunk,
                        "filename": filename,
                        "page": page,
                    }
                )

            if end >= len(text):
                break

            start += step

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(
    chunks,
):

    if not chunks:

        return np.empty(
            (0, 384),
            dtype="float32",
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype="float32",
    )


# ============================================================
# FAISS INDEX
# ============================================================

def create_faiss_index(
    embeddings,
):

    if embeddings is None:
        return None

    if len(embeddings) == 0:
        return None

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(
        embeddings
    )

    return index


# ============================================================
# KEYWORD TOKENIZATION
# ============================================================

STOP_WORDS = {
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
    "who",
    "does",
    "do",
    "can",
    "could",
    "would",
    "should",
    "please",
    "tell",
    "me",
}


def tokenize(text):

    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower(),
    )

    return [
        word
        for word in words
        if word not in STOP_WORDS
        and len(word) > 2
    ]


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    question,
    chunk_text,
):

    question_words = set(
        tokenize(question)
    )

    chunk_words = set(
        tokenize(chunk_text)
    )

    if not question_words:
        return 0.0

    matches = (
        question_words
        .intersection(chunk_words)
    )

    return (
        len(matches)
        /
        len(question_words)
    )


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    question,
    top_k=TOP_K,
):

    chunks = st.session_state.chunks

    index = st.session_state.faiss_index

    if not chunks:
        return []

    if index is None:
        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    semantic_k = min(
        max(top_k * 3, top_k),
        len(chunks),
    )

    semantic_scores, semantic_indices = (
        index.search(
            question_embedding,
            semantic_k,
        )
    )

    semantic_scores = semantic_scores[0]

    semantic_indices = semantic_indices[0]

    semantic_results = {}

    for score, index_number in zip(
        semantic_scores,
        semantic_indices,
    ):

        if index_number < 0:
            continue

        semantic_results[
            int(index_number)
        ] = float(score)

    results = []

    for index_number, chunk in enumerate(
        chunks
    ):

        semantic_score = semantic_results.get(
            index_number,
            0.0,
        )

        keyword = keyword_score(
            question,
            chunk["text"],
        )

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
                "hybrid_score": hybrid_score,
            }
        )

    results.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return results[:top_k]


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key():

    # First try Streamlit secrets.
    try:

        api_key = st.secrets.get(
            "GROQ_API_KEY"
        )

        if api_key:

            return str(api_key).strip()

    except Exception:
        pass

    # Environment variable fallback.
    # Still never hardcoded.
    api_key = os.environ.get(
        "GROQ_API_KEY"
    )

    if api_key:

        return api_key.strip()

    return None


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

    api_key = get_groq_api_key()

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
    retrieved_chunks,
):

    client = get_groq_client()

    if client is None:

        return (
            "GROQ_API_KEY is not configured.\n\n"
            "Add the following secret to your Streamlit "
            "app settings:\n\n"
            "GROQ_API_KEY = \"your_api_key\""
        )

    if not retrieved_chunks:

        return (
            "The information is not available in the "
            "provided documents."
        )

    context_parts = []

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):

        if chunk["page"] is not None:

            page_text = (
                f", Page {chunk['page']}"
            )

        else:

            page_text = ""

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
You are an AI Document Assistant.

Your job is to answer questions using ONLY
the document context supplied by the user.

STRICT RULES:

1. Use only the provided document context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. Do not assume information that is not present.
5. If the answer cannot be found in the context,
   respond:

"The information is not available in the
provided documents."

6. Keep the answer clear and useful.
7. When possible, mention the relevant policy,
   document, section, or page based only on the
   supplied context.
"""

    user_prompt = f"""
DOCUMENT CONTEXT
================

{context}

USER QUESTION
=============

{question}
"""

    try:

        response = client.chat.completions.create(
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
            temperature=0,
            max_completion_tokens=700,
        )

        if not response.choices:

            return (
                "The AI model returned no answer."
            )

        answer = response.choices[0].message.content

        if not answer:

            return (
                "The AI model returned an empty answer."
            )

        return answer.strip()

    except AuthenticationError:

        return (
            "The GROQ_API_KEY is invalid or not authorized. "
            "Please update the GROQ_API_KEY in Streamlit "
            "Secrets."
        )

    except APIStatusError as error:

        if getattr(error, "status_code", None) == 404:

            return (
                f"The configured Groq model "
                f"'{GROQ_MODEL}' is unavailable for this "
                f"API project. Please check the Groq model "
                f"permissions."
            )

        if getattr(error, "status_code", None) == 429:

            return (
                "Groq rate limit reached. Please wait a "
                "moment and try again."
            )

        return (
            "Groq returned an API error while generating "
            "the answer."
        )

    except APIConnectionError:

        return (
            "Could not connect to Groq. Please try again."
        )

    except Exception as error:

        return (
            "An unexpected error occurred while generating "
            "the answer.\n\n"
            f"Details: {error}"
        )


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

def process_documents(
    documents,
):

    if not documents:
        return False

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

        if embeddings.size == 0:

            return False

        index = create_faiss_index(
            embeddings
        )

        if index is None:

            return False

    st.session_state.documents = documents

    st.session_state.chunks = chunks

    st.session_state.embeddings = embeddings

    st.session_state.faiss_index = index

    st.session_state.processed = True

    return True


# ============================================================
# GOOGLE DRIVE DOWNLOAD
# ============================================================

def load_from_google_drive(
    url,
):

    url = url.strip()

    if not url:
        return []

    temp_directory = tempfile.mkdtemp()

    try:

        # ----------------------------------------------------
        # GOOGLE DRIVE FOLDER
        # ----------------------------------------------------

        if "/folders/" in url:

            downloaded_files = (
                gdown.download_folder(
                    url=url,
                    output=temp_directory,
                    quiet=True,
                    use_cookies=False,
                )
            )

            if not downloaded_files:

                return []

            file_paths = []

            for item in downloaded_files:

                path = Path(item)

                if (
                    path.is_file()
                    and
                    path.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                ):

                    file_paths.append(
                        path
                    )

        # ----------------------------------------------------
        # GOOGLE DRIVE FILE
        # ----------------------------------------------------

        else:

            downloaded_path = gdown.download(
                url=url,
                output=temp_directory,
                quiet=True,
                fuzzy=True,
                use_cookies=False,
            )

            if not downloaded_path:

                return []

            downloaded_path = Path(
                downloaded_path
            )

            if downloaded_path.is_file():

                file_paths = [
                    downloaded_path
                ]

            else:

                file_paths = [
                    path
                    for path in downloaded_path.rglob("*")
                    if (
                        path.is_file()
                        and
                        path.suffix.lower()
                        in SUPPORTED_EXTENSIONS
                    )
                ]

        # ----------------------------------------------------
        # EXTRACT
        # ----------------------------------------------------

        documents = []

        for file_path in file_paths:

            extension = (
                file_path.suffix.lower()
            )

            if extension not in SUPPORTED_EXTENSIONS:
                continue

            filename = file_path.name

            try:

                extracted = extract_document(
                    str(file_path),
                    filename,
                )

                documents.extend(
                    extracted
                )

            except Exception as error:

                st.warning(
                    f"Could not process "
                    f"'{filename}': {error}"
                )

        return documents

    except Exception as error:

        st.error(
            "Could not load the Google Drive source."
        )

        st.caption(
            f"Details: {error}"
        )

        return []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-title">'
    'Document Sources'
    '</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    '<div class="sidebar-description">'
    'Upload local documents or connect a public '
    'Google Drive file or folder.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# LOCAL UPLOAD
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-heading">'
    '📁 Local Documents'
    '</div>',
    unsafe_allow_html=True,
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT or MD files",
    type=[
        "pdf",
        "docx",
        "txt",
        "md",
    ],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:

    st.sidebar.success(
        f"{len(uploaded_files)} file(s) selected"
    )


# ============================================================
# GOOGLE DRIVE
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-heading">'
    '☁️ Google Drive'
    '</div>',
    unsafe_allow_html=True,
)

drive_url = st.sidebar.text_input(
    "Google Drive URL",
    placeholder="Paste a public Drive file or folder link",
)

st.sidebar.caption(
    "Supported: PDF • DOCX • TXT • MD"
)


# ============================================================
# PROCESS BUTTON
# ============================================================

st.sidebar.markdown("")

process_button = st.sidebar.button(
    "⚡ Process Documents",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESS LOCAL + DRIVE DOCUMENTS
# ============================================================

if process_button:

    all_documents = []

    processing_errors = []

    # --------------------------------------------------------
    # LOCAL UPLOADS
    # --------------------------------------------------------

    if uploaded_files:

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            extension = Path(
                filename
            ).suffix.lower()

            if extension not in SUPPORTED_EXTENSIONS:

                processing_errors.append(
                    f"{filename}: unsupported file type"
                )

                continue

            try:

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=extension,
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_path = temp_file.name

                try:

                    extracted = extract_document(
                        temp_path,
                        filename,
                    )

                    all_documents.extend(
                        extracted
                    )

                finally:

                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

            except Exception as error:

                processing_errors.append(
                    f"{filename}: {error}"
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
                    drive_url
                )
            )

            all_documents.extend(
                drive_documents
            )

    # --------------------------------------------------------
    # PROCESS EVERYTHING
    # --------------------------------------------------------

    if all_documents:

        try:

            success = process_documents(
                all_documents
            )

            if success:

                st.sidebar.success(
                    "Knowledge base ready."
                )

                st.sidebar.info(
                    f"{len(all_documents)} extracted "
                    f"document section(s)"
                )

            else:

                st.sidebar.error(
                    "Documents were found, but no searchable "
                    "text could be created."
                )

        except Exception as error:

            st.sidebar.error(
                "Document processing failed."
            )

            st.sidebar.caption(
                f"Details: {error}"
            )

    else:

        st.sidebar.warning(
            "No supported document text was found."
        )

    # --------------------------------------------------------
    # PROCESSING WARNINGS
    # --------------------------------------------------------

    for error_message in processing_errors:

        st.sidebar.warning(
            error_message
        )


# ============================================================
# KNOWLEDGE BASE
# ============================================================

if st.session_state.documents:

    unique_documents = {}

    for document in st.session_state.documents:

        filename = document["filename"]

        if filename not in unique_documents:

            unique_documents[filename] = {
                "pages": set(),
                "characters": 0,
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

    # --------------------------------------------------------
    # SECTION HEADER
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">'
        '<div class="section-title">'
        '📄 Knowledge Base'
        '</div>'
        '<div class="section-description">'
        'Your processed documents are ready for semantic '
        'and keyword-based retrieval.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">'
            f'{total_documents}'
            f'</div>'
            f'<div class="metric-label">'
            f'Documents'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">'
            f'{total_chunks}'
            f'</div>'
            f'<div class="metric-label">'
            f'Searchable Chunks'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-value">'
            f'{total_characters:,}'
            f'</div>'
            f'<div class="metric-label">'
            f'Characters'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
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

        safe_filename = html.escape(
            filename
        )

        st.markdown(
            f'<div class="document-card">'
            f'<div class="document-name">'
            f'📄 {safe_filename}'
            f'</div>'
            f'<div class="document-meta">'
            f'{page_info} &nbsp;•&nbsp; '
            f'{information["characters"]:,} characters'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# ASK YOUR DOCUMENTS
# ============================================================

st.markdown(
    '<div class="section">'
    '<div class="section-title">'
    '💬 Ask Your Documents'
    '</div>'
    '<div class="section-description">'
    'Ask a question about your processed documents. '
    'The assistant retrieves relevant passages before '
    'generating an answer.'
    '</div>',
    unsafe_allow_html=True,
)

question = st.text_input(
    "Question",
    placeholder="What is the annual leave policy?",
    label_visibility="collapsed",
)

ask_button = st.button(
    "🔍 Ask Assistant",
    type="primary",
    use_container_width=True,
)

st.markdown(
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# ANSWER QUESTION
# ============================================================

if ask_button:

    # --------------------------------------------------------
    # NO DOCUMENTS
    # --------------------------------------------------------

    if not st.session_state.processed:

        st.warning(
            "Please upload and process documents first."
        )

    # --------------------------------------------------------
    # EMPTY QUESTION
    # --------------------------------------------------------

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    # --------------------------------------------------------
    # ASK
    # --------------------------------------------------------

    else:

        with st.spinner(
            "Searching your knowledge base..."
        ):

            retrieved_chunks = hybrid_search(
                question,
                top_k=TOP_K,
            )

        if not retrieved_chunks:

            st.warning(
                "No relevant information was found "
                "in the processed documents."
            )

        else:

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_chunks,
                )

            # ------------------------------------------------
            # ESCAPE HTML CONTENT
            # ------------------------------------------------

            safe_question = html.escape(
                question
            )

            safe_answer = html.escape(
                answer
            )

            # ------------------------------------------------
            # USER MESSAGE
            # ------------------------------------------------

            st.markdown(
                f'<div class="user-message">'
                f'<div class="user-label">YOU</div>'
                f'<div class="user-text">'
                f'{safe_question}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # AI ANSWER
            # ------------------------------------------------

            st.markdown(
                f'<div class="ai-message">'
                f'<div class="ai-label">'
                f'🤖 AI DOCUMENT ASSISTANT'
                f'</div>'
                f'<div class="ai-text">'
                f'{safe_answer}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            st.markdown(
                '<div class="sources-title">'
                '📚 Retrieved Sources'
                '</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<div class="sources-description">'
                f'{len(retrieved_chunks)} relevant chunks '
                f'were retrieved for this answer.'
                f'</div>',
                unsafe_allow_html=True,
            )

            for i, source in enumerate(
                retrieved_chunks,
                start=1,
            ):

                if source["page"] is not None:

                    page_text = (
                        f"Page {source['page']}"
                    )

                else:

                    page_text = (
                        "Page information unavailable"
                    )

                safe_filename = html.escape(
                    source["filename"]
                )

                safe_text = html.escape(
                    source["text"]
                )

                with st.expander(
                    f"Source {i}  •  "
                    f"{source['filename']}  •  "
                    f"{page_text}"
                ):

                    st.markdown(
                        f'<div class="source-box">'
                        f'<div class="source-file">'
                        f'📄 {safe_filename}'
                        f'</div>'
                        f'<div class="source-page">'
                        f'{page_text}'
                        f'</div>'
                        f'<div class="source-text">'
                        f'{safe_text}'
                        f'</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    st.caption(
                        f"Semantic: "
                        f"{source['semantic_score']:.3f}"
                        f"  •  "
                        f"Keyword: "
                        f"{source['keyword_score']:.3f}"
                        f"  •  "
                        f"Hybrid: "
                        f"{source['hybrid_score']:.3f}"
                    )
