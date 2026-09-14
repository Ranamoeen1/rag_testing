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
GROQ_MODEL = "llama-3.1-8b-instant"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Document Assistant")
st.write(
    "Upload documents or load them from Google Drive, then ask questions "
    "about their content."
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
    Keeps page number for every extracted page.
    """

    documents = []

    reader = PdfReader(file_path)

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text() or ""

        if text.strip():
            documents.append({
                "text": text,
                "filename": filename,
                "page": page_number
            })

    return documents


def extract_docx(file_path, filename):
    """
    Extract text from DOCX.
    DOCX does not provide reliable page numbers here,
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

    return [{
        "text": full_text,
        "filename": filename,
        "page": None
    }]


def extract_txt(file_path, filename):
    """
    Extract text from TXT.
    """

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()

    if not text.strip():
        return []

    return [{
        "text": text,
        "filename": filename,
        "page": None
    }]


def extract_md(file_path, filename):
    """
    Extract text from Markdown.
    """

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        text = file.read()

    if not text.strip():
        return []

    return [{
        "text": text,
        "filename": filename,
        "page": None
    }]


def extract_document(file_path, filename):
    """
    Select the correct extraction function based on extension.
    """

    extension = Path(filename).suffix.lower()

    if extension == ".pdf":
        return extract_pdf(file_path, filename)

    elif extension == ".docx":
        return extract_docx(file_path, filename)

    elif extension == ".txt":
        return extract_txt(file_path, filename)

    elif extension == ".md":
        return extract_md(file_path, filename)

    return []


# ============================================================
# CHUNKING
# ============================================================

def chunk_text(documents):
    """
    Split extracted documents into overlapping chunks.

    Metadata such as filename and page is preserved.
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

                chunks.append({
                    "text": chunk,
                    "filename": filename,
                    "page": page
                })

            start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

def create_embeddings(chunks):
    """
    Create embeddings once for all chunks.
    """

    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return embeddings.astype("float32")


# ============================================================
# FAISS INDEX
# ============================================================

def create_faiss_index(embeddings):
    """
    Create a FAISS cosine-similarity index.

    Because embeddings are normalized, inner product
    behaves like cosine similarity.
    """

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    return index


# ============================================================
# KEYWORD SEARCH
# ============================================================

def tokenize(text):
    """
    Simple keyword tokenizer.
    """

    words = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    # Remove very common words
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
        if word not in stop_words and len(word) > 2
    ]


def keyword_score(question, chunk_text):
    """
    Score a chunk based on matching question keywords.
    """

    question_words = set(tokenize(question))
    chunk_words = set(tokenize(chunk_text))

    if not question_words:
        return 0.0

    matches = question_words.intersection(chunk_words)

    return len(matches) / len(question_words)


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(question, top_k=TOP_K):
    """
    Combine semantic FAISS search and keyword search.
    """

    chunks = st.session_state.chunks
    index = st.session_state.faiss_index

    if not chunks or index is None:
        return []

    # --------------------------------------------------------
    # Semantic search
    # --------------------------------------------------------

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    semantic_scores, semantic_indices = index.search(
        question_embedding,
        min(top_k * 2, len(chunks))
    )

    semantic_scores = semantic_scores[0]
    semantic_indices = semantic_indices[0]

    semantic_results = {}

    for score, index_number in zip(
        semantic_scores,
        semantic_indices
    ):

        semantic_results[int(index_number)] = float(score)

    # --------------------------------------------------------
    # Keyword search
    # --------------------------------------------------------

    results = []

    for index_number, chunk in enumerate(chunks):

        semantic_score = semantic_results.get(
            index_number,
            0.0
        )

        keyword = keyword_score(
            question,
            chunk["text"]
        )

        # Hybrid score
        hybrid_score = (
            0.75 * semantic_score +
            0.25 * keyword
        )

        results.append({
            "text": chunk["text"],
            "filename": chunk["filename"],
            "page": chunk["page"],
            "semantic_score": semantic_score,
            "keyword_score": keyword,
            "hybrid_score": hybrid_score
        })

    # Highest score first
    results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# GROQ
# ============================================================

def get_groq_client():
    """
    Read GROQ_API_KEY from Streamlit secrets.
    """

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        return None

    return Groq(api_key=api_key)


def generate_answer(question, retrieved_chunks):
    """
    Ask Groq to answer only from retrieved context.
    """

    client = get_groq_client()

    if client is None:
        return (
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit secrets."
        )

    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):

        page_text = ""

        if chunk["page"] is not None:
            page_text = f", Page {chunk['page']}"

        context_parts.append(
            f"""
SOURCE {i}
Filename: {chunk['filename']}{page_text}

Content:
{chunk['text']}
"""
        )

    context = "\n".join(context_parts)

    system_prompt = """
You are an AI document assistant.

Answer the user's question ONLY using the provided document context.

Rules:
1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not available in the context, clearly say:
   "The information is not available in the provided documents."
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

    chunks = chunk_text(documents)

    if not chunks:
        return False

    with st.spinner("Creating document embeddings..."):

        embeddings = create_embeddings(chunks)

        index = create_faiss_index(embeddings)

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
    Download a public Google Drive file or folder.

    gdown handles Drive share links.
    """

    temp_directory = tempfile.mkdtemp()

    try:

        # Detect folder URL
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

        else:

            downloaded_path = gdown.download(
                url=url,
                output=temp_directory,
                quiet=True
            )

            if not downloaded_path:
                return []

            downloaded_path = Path(downloaded_path)

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

                file_paths = [downloaded_path]

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

            documents.extend(extracted)

        return documents

    except Exception as e:

        st.error(
            f"Could not load Google Drive content: {e}"
        )

        return []


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("📄 Document Sources")


uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)


st.sidebar.markdown("---")

st.sidebar.subheader("Google Drive")

drive_url = st.sidebar.text_input(
    "Paste a public Drive file or folder link"
)


process_button = st.sidebar.button(
    "Process Documents",
    use_container_width=True
)


# ============================================================
# PROCESS BUTTON
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

            all_documents.extend(extracted)

    # --------------------------------------------------------
    # GOOGLE DRIVE
    # --------------------------------------------------------

    if drive_url.strip():

        with st.spinner(
            "Loading documents from Google Drive..."
        ):

            drive_documents = load_from_google_drive(
                drive_url.strip()
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

    st.subheader("📋 Extracted Document Information")

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
            unique_documents[filename]["pages"].add(
                page
            )

        unique_documents[filename]["characters"] += len(
            document["text"]
        )

    for filename, information in unique_documents.items():

        pages = information["pages"]

        if pages:
            page_info = f"{len(pages)} page(s)"
        else:
            page_info = "Page information unavailable"

        st.write(
            f"**{filename}** — "
            f"{page_info} — "
            f"{information['characters']:,} characters"
        )

    st.info(
        f"Created **{len(st.session_state.chunks)} chunks** "
        f"from the loaded documents."
    )


# ============================================================
# QUESTION ANSWERING
# ============================================================

st.subheader("💬 Ask a Question")

question = st.text_input(
    "Enter your question",
    placeholder="What does the document say about annual leave?"
)


if st.button("Ask"):

    if not st.session_state.processed:

        st.warning(
            "Please upload or load documents and "
            "click 'Process Documents' first."
        )

    elif not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        with st.spinner("Searching documents..."):

            retrieved_chunks = hybrid_search(
                question,
                top_k=TOP_K
            )

        if not retrieved_chunks:

            st.warning(
                "No relevant information was found."
            )

        else:

            with st.spinner("Generating answer..."):

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            # ------------------------------------------------
            # ANSWER
            # ------------------------------------------------

            st.subheader("🤖 Answer")

            st.write(answer)

            # ------------------------------------------------
            # SOURCES
            # ------------------------------------------------

            st.subheader("📚 Retrieved Sources")

            for i, source in enumerate(
                retrieved_chunks,
                start=1
            ):

                page_text = ""

                if source["page"] is not None:
                    page_text = (
                        f" | Page {source['page']}"
                    )

                with st.expander(
                    f"Source {i}: "
                    f"{source['filename']}"
                    f"{page_text}"
                ):

                    st.write(
                        source["text"]
                    )

                    st.caption(
                        f"Semantic score: "
                        f"{source['semantic_score']:.3f} | "
                        f"Keyword score: "
                        f"{source['keyword_score']:.3f} | "
                        f"Hybrid score: "
                        f"{source['hybrid_score']:.3f}"
                    )
