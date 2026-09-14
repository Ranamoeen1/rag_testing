import os
import re
import html
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

# Groq model
GROQ_MODEL = "openai/gpt-oss-20b"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
<style>
/* ==========================================================
   GLOBAL
   ========================================================== */

.stApp {
    background: #f6f8fb;
}

.main .block-container {
    max-width: 1180px;
    padding-top: 35px;
    padding-bottom: 60px;
}


/* ==========================================================
   HEADER
   ========================================================== */

.app-header {
    background: #ffffff;
    border: 1px solid #e4e8ef;
    border-radius: 18px;
    padding: 30px 34px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(15, 23, 42, 0.04);
}

.app-header-title {
    font-size: 31px;
    font-weight: 750;
    color: #172033;
    line-height: 1.2;
    margin: 0;
}

.app-header-subtitle {
    color: #6b7280;
    font-size: 15px;
    line-height: 1.6;
    margin-top: 9px;
}


/* ==========================================================
   SECTION
   ========================================================== */

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


/* ==========================================================
   METRICS
   ========================================================== */

.metric-card {
    background: #f9fafc;
    border: 1px solid #e5e9ef;
    border-radius: 13px;
    padding: 18px 12px;
    text-align: center;
    min-height: 92px;
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


/* ==========================================================
   DOCUMENTS
   ========================================================== */

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


/* ==========================================================
   CHAT
   ========================================================== */

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


/* ==========================================================
   SOURCES
   ========================================================== */

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
}


/* ==========================================================
   SIDEBAR
   ========================================================== */

section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #e4e8ef;
}

section[data-testid="stSidebar"] .block-container {
    padding-top: 30px;
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


/* ==========================================================
   UPLOADER
   ========================================================== */

[data-testid="stFileUploader"] {
    background: #fafbfc;
    border: 1px dashed #cbd3df;
    border-radius: 12px;
    padding: 7px;
}


/* ==========================================================
   INPUTS
   ========================================================== */

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


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {
    border-radius: 10px;
    min-height: 43px;
    font-weight: 650;
    border: 1px solid #d7dde6;
}

.stButton > button:hover {
    border-color: #9aa6b7;
}


/* ==========================================================
   EXPANDERS
   ========================================================== */

[data-testid="stExpander"] {
    background: #ffffff;
    border: 1px solid #e1e6ed;
    border-radius: 11px;
    margin-bottom: 8px;
}


/* ==========================================================
   STATUS
   ========================================================== */

.status-success {
    display: inline-block;
    background: #edf8f1;
    border: 1px solid #d4eedc;
    color: #287344;
    border-radius: 20px;
    padding: 5px 11px;
    font-size: 12px;
    font-weight: 650;
}


/* ==========================================================
   HIDE UNNECESSARY ELEMENTS
   ========================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HERO HEADER
# ============================================================

st.markdown(
    '<div class="app-header">'
    '<div class="app-header-title">'
    '📚 AI Document Assistant'
    '</div>'
    '<div class="app-header-subtitle">'
    'Upload your documents, build a searchable knowledge base, '
    'and ask questions using AI-powered document retrieval.'
    '</div>'
    '</div>',
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
# EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


embedding_model = load_embedding_model()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(file_path, filename):

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


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx(file_path, filename):

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


# ============================================================
# TXT EXTRACTION
# ============================================================

def extract_txt(file_path, filename):

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


# ============================================================
# MARKDOWN EXTRACTION
# ============================================================

def extract_md(file_path, filename):

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


# ============================================================
# DOCUMENT EXTRACTION ROUTER
# ============================================================

def extract_document(file_path, filename):

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":

        return extract_pdf(
            file_path,
            filename
        )

    if extension == ".docx":

        return extract_docx(
            file_path,
            filename
        )

    if extension == ".txt":

        return extract_txt(
            file_path,
            filename
        )

    if extension == ".md":

        return extract_md(
            file_path,
            filename
        )

    return []


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(documents):

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
# EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

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
# FAISS
# ============================================================

def create_faiss_index(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(
        dimension
    )

    index.add(embeddings)

    return index


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):

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


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(
    question,
    chunk_text
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
        question_words.intersection(
            chunk_words
        )
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

    chunks = st.session_state.chunks

    index = st.session_state.faiss_index

    if not chunks or index is None:
        return []

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
        key=lambda item: item["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# GROQ CLIENT
# ============================================================

def get_groq_client():

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

    client = get_groq_client()

    if client is None:

        return (
            "GROQ_API_KEY is not configured. "
            "Please add GROQ_API_KEY to your "
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

1. Use only the provided context.
2. Do not use outside knowledge.
3. Do not invent information.
4. If the answer is not available in the context,
   say exactly:

"The information is not available in the
provided documents."

5. Keep answers clear and concise.
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

    chunks = chunk_text(
        documents
    )

    if not chunks:
        return False

    with st.spinner(
        "Building your document knowledge base..."
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

    temp_directory = tempfile.mkdtemp()

    try:

        # ----------------------------------------------------
        # FOLDER
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
        # FILE
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
        # EXTRACTION
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

    except Exception as error:

        st.error(
            f"Could not load Google Drive content: {error}"
        )

        return []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-title">'
    'Document Sources'
    '</div>',
    unsafe_allow_html=True
)

st.sidebar.markdown(
    '<div class="sidebar-description">'
    'Upload files or connect a public Google Drive '
    'source to build your knowledge base.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOCAL UPLOAD
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-heading">'
    '📁 Local Documents'
    '</div>',
    unsafe_allow_html=True
)

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, TXT or MD",
    type=[
        "pdf",
        "docx",
        "txt",
        "md"
    ],
    accept_multiple_files=True,
    label_visibility="collapsed"
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
    unsafe_allow_html=True
)

drive_url = st.sidebar.text_input(
    "Google Drive URL",
    placeholder="Paste a public Drive file or folder link"
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
    use_container_width=True
)


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if process_button:

    all_documents = []

    # --------------------------------------------------------
    # LOCAL FILES
    # --------------------------------------------------------

    if uploaded_files:

        for uploaded_file in uploaded_files:

            suffix = Path(
                uploaded_file.name
            ).suffix.lower()

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
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
    # PROCESS
    # --------------------------------------------------------

    if all_documents:

        success = process_documents(
            all_documents
        )

        if success:

            st.sidebar.success(
                "Knowledge base ready."
            )

    else:

        st.sidebar.warning(
            "No supported documents found."
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
    # SECTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section">'
        '<div class="section-title">'
        '📄 Knowledge Base'
        '</div>'
        '<div class="section-description">'
        'Your processed documents are ready for semantic '
        'and keyword-based search.'
        '</div>',
        unsafe_allow_html=True
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
            unsafe_allow_html=True
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
            unsafe_allow_html=True
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
            unsafe_allow_html=True
        )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# ASK DOCUMENTS
# ============================================================

st.markdown(
    '<div class="section">'
    '<div class="section-title">'
    '💬 Ask Your Documents'
    '</div>'
    '<div class="section-description">'
    'Ask a question about your uploaded documents. '
    'The assistant retrieves the most relevant passages '
    'before generating an answer.'
    '</div>',
    unsafe_allow_html=True
)


question = st.text_input(
    "Question",
    placeholder="What is the annual leave policy?",
    label_visibility="collapsed"
)


ask_button = st.button(
    "🔍 Ask Assistant",
    type="primary",
    use_container_width=True
)


st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# ANSWER QUESTION
# ============================================================

if ask_button:

    if not st.session_state.processed:

        st.warning(
            "Please upload and process documents first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ----------------------------------------------------
        # SEARCH
        # ----------------------------------------------------

        with st.spinner(
            "Searching your knowledge base..."
        ):

            retrieved_chunks = hybrid_search(
                question,
                top_k=TOP_K
            )


        if not retrieved_chunks:

            st.warning(
                "No relevant information was found "
                "in the processed documents."
            )

        else:

            # ------------------------------------------------
            # GENERATE
            # ------------------------------------------------

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )


            # ------------------------------------------------
            # SAFE HTML
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
                f'<div class="user-label">'
                f'YOU'
                f'</div>'
                f'<div class="user-text">'
                f'{safe_question}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # AI MESSAGE
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
                unsafe_allow_html=True
            )


            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            st.markdown(
                '<div class="sources-title">'
                '📚 Retrieved Sources'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="sources-description">'
                f'{len(retrieved_chunks)} relevant chunks '
                f'used to generate this answer.'
                f'</div>',
                unsafe_allow_html=True
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
