import io
import os
import re
import hashlib
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs

import numpy as np
import requests
import streamlit as st
import faiss

from docx import Document as DocxDocument
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".markdown"}

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Use a currently available Groq model through the Groq API.
# This can also be overridden through Streamlit secrets.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

SEMANTIC_TOP_K = 8
KEYWORD_TOP_K = 8
FINAL_TOP_K = 6

# Hybrid weighting
SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .main {
        background-color: #f8fafc;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Header */
    .app-header {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #111827 0%,
            #1e293b 100%
        );
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
    }

    .app-header h1 {
        margin: 0;
        font-size: 2rem;
        font-weight: 750;
        letter-spacing: -0.03em;
    }

    .app-header p {
        margin: 0.55rem 0 0;
        color: #cbd5e1;
        font-size: 1rem;
    }

    /* Cards */
    .info-card {
        padding: 1rem 1.2rem;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        background: white;
        margin-bottom: 1rem;
    }

    /* Metrics */
    .metric-card {
        padding: 1rem;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        text-align: center;
    }

    .metric-number {
        font-size: 1.5rem;
        font-weight: 750;
        color: #0f172a;
    }

    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.7rem;
        border: 1px solid #cbd5e1;
        transition: all 0.15s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: #64748b;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border-radius: 15px;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        border-radius: 15px;
        margin-bottom: 0.75rem;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 650;
    }

    /* Source cards */
    .source-card {
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.9rem;
        margin-bottom: 0.7rem;
        background: #ffffff;
    }

    .source-title {
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.3rem;
    }

    .source-meta {
        color: #64748b;
        font-size: 0.82rem;
        margin-bottom: 0.5rem;
    }

    .source-text {
        color: #334155;
        font-size: 0.9rem;
        line-height: 1.55;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

def initialize_session_state():
    """Initialize all persistent application state."""

    defaults = {
        "documents": [],
        "chunks": [],
        "embeddings": None,
        "faiss_index": None,
        "tfidf_vectorizer": None,
        "tfidf_matrix": None,
        "processed_hash": None,
        "chat_history": [],
        "drive_files": [],
        "processed_source": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ============================================================
# SECRET / CONFIG HELPERS
# ============================================================

def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    """
    Read a secret from Streamlit secrets first, then environment variables.
    """

    try:
        if name in st.secrets:
            value = st.secrets[name]
            if value:
                return str(value)
    except Exception:
        pass

    value = os.getenv(name)

    if value:
        return value

    return default


def get_groq_api_key() -> Optional[str]:
    return get_secret("GROQ_API_KEY")


def get_drive_api_key() -> Optional[str]:
    """
    Optional.

    This is only needed when enumerating files inside a public Google Drive
    folder using the Google Drive API.

    Individual public files can be downloaded without it.
    """
    return get_secret("GOOGLE_DRIVE_API_KEY")


def get_groq_model() -> str:
    return get_secret(
        "GROQ_MODEL",
        DEFAULT_GROQ_MODEL
    )


# ============================================================
# FILE HELPERS
# ============================================================

def get_extension(filename: str) -> str:
    return os.path.splitext(filename.lower())[1]


def is_supported_file(filename: str) -> bool:
    return get_extension(filename) in SUPPORTED_EXTENSIONS


def calculate_bytes_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ============================================================
# DOCUMENT EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Extract PDF text page-by-page.

    Page numbers are preserved in metadata.
    """

    results = []

    try:
        reader = PdfReader(io.BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            text = text.strip()

            if text:
                results.append(
                    {
                        "text": text,
                        "metadata": {
                            "filename": filename,
                            "file_type": "pdf",
                            "page": page_number,
                        },
                    }
                )

    except Exception as exc:
        raise ValueError(
            f"Failed to extract PDF '{filename}': {exc}"
        )

    return results


def extract_docx_text(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Extract text from DOCX paragraphs.
    """

    try:
        document = DocxDocument(io.BytesIO(file_bytes))

        paragraphs = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        text = "\n".join(paragraphs).strip()

        if not text:
            return []

        return [
            {
                "text": text,
                "metadata": {
                    "filename": filename,
                    "file_type": "docx",
                    "page": None,
                },
            }
        ]

    except Exception as exc:
        raise ValueError(
            f"Failed to extract DOCX '{filename}': {exc}"
        )


def extract_text_file(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Extract TXT / Markdown content.
    """

    try:
        text = file_bytes.decode("utf-8", errors="replace").strip()

        if not text:
            return []

        extension = get_extension(filename)

        return [
            {
                "text": text,
                "metadata": {
                    "filename": filename,
                    "file_type": extension.lstrip("."),
                    "page": None,
                },
            }
        ]

    except Exception as exc:
        raise ValueError(
            f"Failed to read text file '{filename}': {exc}"
        )


def extract_document(file_bytes: bytes, filename: str) -> List[Dict]:
    """
    Automatically select the correct document parser.
    """

    extension = get_extension(filename)

    if extension == ".pdf":
        return extract_pdf_text(file_bytes, filename)

    if extension == ".docx":
        return extract_docx_text(file_bytes, filename)

    if extension in {".txt", ".md", ".markdown"}:
        return extract_text_file(file_bytes, filename)

    raise ValueError(
        f"Unsupported file type: {filename}"
    )


# ============================================================
# CHUNKING
# ============================================================

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Simple overlapping word-based chunking.

    Keeping this implementation straightforward makes the RAG
    pipeline easy to explain during teaching/presentation.
    """

    words = text.split()

    if not words:
        return []

    chunks = []

    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap

    return chunks


def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """
    Split extracted documents into overlapping chunks while
    preserving the original metadata.
    """

    chunks = []

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        text_chunks = split_text(text)

        for chunk_id, chunk_text in enumerate(text_chunks):

            chunk_metadata = dict(metadata)

            chunk_metadata["chunk_id"] = chunk_id

            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": chunk_metadata,
                }
            )

    return chunks


# ============================================================
# EMBEDDINGS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """
    Load the local Sentence Transformers embedding model once.
    """

    return SentenceTransformer(DEFAULT_EMBEDDING_MODEL)


def create_embeddings(texts: List[str]) -> np.ndarray:
    """
    Generate local embeddings.

    Groq is NOT used for embeddings.
    """

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return np.asarray(
        embeddings,
        dtype="float32",
    )


# ============================================================
# SEARCH INDEX
# ============================================================

def build_search_index(chunks: List[Dict]):
    """
    Build both:

    1. FAISS semantic/vector index
    2. TF-IDF keyword index
    """

    if not chunks:
        raise ValueError("No chunks available for indexing.")

    texts = [chunk["text"] for chunk in chunks]

    # Create semantic embeddings.
    embeddings = create_embeddings(texts)

    # FAISS inner-product search works as cosine similarity
    # because embeddings are normalized.
    dimension = embeddings.shape[1]

    faiss_index = faiss.IndexFlatIP(dimension)

    faiss_index.add(embeddings)

    # TF-IDF provides keyword-oriented retrieval.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return (
        embeddings,
        faiss_index,
        vectorizer,
        tfidf_matrix,
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    top_k: int = SEMANTIC_TOP_K,
) -> List[Tuple[int, float]]:
    """
    Semantic search using FAISS.
    """

    if st.session_state.faiss_index is None:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = st.session_state.faiss_index.search(
        query_embedding,
        min(
            top_k,
            st.session_state.faiss_index.ntotal,
        ),
    )

    results = []

    for index, score in zip(indices[0], scores[0]):

        if index < 0:
            continue

        results.append(
            (
                int(index),
                float(score),
            )
        )

    return results


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(
    query: str,
    top_k: int = KEYWORD_TOP_K,
) -> List[Tuple[int, float]]:
    """
    Keyword search using TF-IDF cosine similarity.
    """

    vectorizer = st.session_state.tfidf_vectorizer
    tfidf_matrix = st.session_state.tfidf_matrix

    if vectorizer is None or tfidf_matrix is None:
        return []

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        tfidf_matrix,
    )[0]

    ranked_indices = np.argsort(
        similarities
    )[::-1]

    results = []

    for index in ranked_indices[:top_k]:

        score = float(similarities[index])

        if score <= 0:
            continue

        results.append(
            (
                int(index),
                score,
            )
        )

    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(
    query: str,
    top_k: int = FINAL_TOP_K,
) -> List[Dict]:
    """
    Combine semantic and keyword search.

    Hybrid score:

        0.65 * semantic_score
        +
        0.35 * keyword_score

    This makes the retrieval strategy easy to explain.
    """

    semantic_results = semantic_search(query)

    keyword_results = keyword_search(query)

    combined_scores = {}

    # Add semantic scores.
    for index, score in semantic_results:

        combined_scores.setdefault(
            index,
            {
                "semantic_score": 0.0,
                "keyword_score": 0.0,
            },
        )

        combined_scores[index]["semantic_score"] = score

    # Add keyword scores.
    for index, score in keyword_results:

        combined_scores.setdefault(
            index,
            {
                "semantic_score": 0.0,
                "keyword_score": 0.0,
            },
        )

        combined_scores[index]["keyword_score"] = score

    ranked = []

    for index, scores in combined_scores.items():

        hybrid_score = (
            SEMANTIC_WEIGHT * scores["semantic_score"]
            +
            KEYWORD_WEIGHT * scores["keyword_score"]
        )

        chunk = st.session_state.chunks[index]

        ranked.append(
            {
                "index": index,
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "semantic_score": scores["semantic_score"],
                "keyword_score": scores["keyword_score"],
                "hybrid_score": hybrid_score,
            }
        )

    ranked.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    return ranked[:top_k]


# ============================================================
# GOOGLE DRIVE HELPERS
# ============================================================

def extract_drive_id(url: str) -> Optional[str]:
    """
    Extract a Google Drive file/folder ID from common Drive URLs.
    """

    url = url.strip()

    parsed = urlparse(url)

    # /file/d/<ID>/view
    match = re.search(
        r"/file/d/([a-zA-Z0-9_-]+)",
        parsed.path,
    )

    if match:
        return match.group(1)

    # /folders/<ID>
    match = re.search(
        r"/folders/([a-zA-Z0-9_-]+)",
        parsed.path,
    )

    if match:
        return match.group(1)

    # ?id=<ID>
    query = parse_qs(parsed.query)

    if "id" in query and query["id"]:
        return query["id"][0]

    return None


def is_google_drive_url(url: str) -> bool:
    try:
        hostname = urlparse(url).hostname or ""

        return (
            hostname == "drive.google.com"
            or hostname.endswith(".drive.google.com")
        )

    except Exception:
        return False


def drive_file_download_url(file_id: str) -> str:
    return (
        "https://drive.google.com/uc"
        f"?export=download&id={file_id}"
    )


def download_public_drive_file(
    file_id: str,
    filename_hint: str = "drive_file",
) -> Tuple[str, bytes]:
    """
    Download a publicly accessible Drive file without OAuth.

    This works for individual files that Drive allows to be downloaded
    publicly.
    """

    url = drive_file_download_url(file_id)

    response = requests.get(
        url,
        timeout=30,
        allow_redirects=True,
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "(compatible; Streamlit RAG application)"
            )
        },
    )

    response.raise_for_status()

    content_type = (
        response.headers.get(
            "Content-Type",
            ""
        ).lower()
    )

    content = response.content

    # A common failure mode is receiving an HTML login/permission page.
    if "text/html" in content_type:

        html_preview = content[:1000].lower()

        if (
            b"sign in" in html_preview
            or b"permission" in html_preview
            or b"access denied" in html_preview
        ):
            raise PermissionError(
                "The Google Drive file is not publicly accessible."
            )

        raise ValueError(
            "Google Drive returned an HTML page instead of a file."
        )

    if not content:
        raise ValueError(
            "The Google Drive file is empty."
        )

    return filename_hint, content


def get_drive_file_metadata(
    file_id: str,
    api_key: str,
) -> Dict:
    """
    Retrieve public file metadata through Drive API v3.

    API key is sufficient for public resources; OAuth is not required
    for public-only access.
    """

    url = (
        f"https://www.googleapis.com/drive/v3/files/{file_id}"
    )

    params = {
        "key": api_key,
        "fields": (
            "id,name,mimeType,size,"
            "resourceKey,webContentLink"
        ),
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    if response.status_code == 404:
        raise FileNotFoundError(
            "Google Drive file was not found or is not publicly accessible."
        )

    if response.status_code in {401, 403}:
        raise PermissionError(
            "Google Drive denied access to this file."
        )

    response.raise_for_status()

    return response.json()


def list_public_drive_folder(
    folder_id: str,
    api_key: str,
) -> List[Dict]:
    """
    List files directly inside a public Google Drive folder.

    Only supported file types are returned.
    """

    url = "https://www.googleapis.com/drive/v3/files"

    files = []

    page_token = None

    while True:

        params = {
            "key": api_key,
            "q": (
                f"'{folder_id}' in parents "
                "and trashed = false"
            ),
            "pageSize": 100,
            "fields": (
                "nextPageToken,"
                "files(id,name,mimeType,size,resourceKey)"
            ),
            "orderBy": "name",
        }

        if page_token:
            params["pageToken"] = page_token

        response = requests.get(
            url,
            params=params,
            timeout=20,
        )

        if response.status_code in {401, 403}:
            raise PermissionError(
                "Google Drive denied access to this folder. "
                "Make sure the folder is shared as "
                "'Anyone with the link'."
            )

        if response.status_code == 404:
            raise FileNotFoundError(
                "Google Drive folder was not found."
            )

        response.raise_for_status()

        data = response.json()

        for item in data.get("files", []):

            name = item.get("name", "")

            if is_supported_file(name):
                files.append(item)

        page_token = data.get("nextPageToken")

        if not page_token:
            break

    return files


def load_drive_source(
    url: str,
) -> Tuple[List[Dict], List[str], List[str]]:
    """
    Load either:

    - one public Drive file
    - a public Drive folder

    Returns:

        documents
        supported_file_names
        skipped_file_names
    """

    if not is_google_drive_url(url):
        raise ValueError(
            "Please enter a valid Google Drive URL."
        )

    drive_id = extract_drive_id(url)

    if not drive_id:
        raise ValueError(
            "Could not extract a Google Drive file/folder ID from the URL."
        )

    # --------------------------------------------------------
    # First try to determine whether this is a single file.
    # This requires the optional API key.
    # --------------------------------------------------------

    api_key = get_drive_api_key()

    documents = []
    supported_names = []
    skipped_names = []

    if api_key:

        try:
            metadata = get_drive_file_metadata(
                drive_id,
                api_key,
            )

            name = metadata.get(
                "name",
                "drive_file",
            )

            mime_type = metadata.get(
                "mimeType",
                "",
            )

            # Google folder
            if mime_type == "application/vnd.google-apps.folder":

                files = list_public_drive_folder(
                    drive_id,
                    api_key,
                )

                for file_info in files:

                    file_id = file_info["id"]
                    filename = file_info["name"]

                    try:
                        _, content = download_public_drive_file(
                            file_id,
                            filename,
                        )

                        extracted = extract_document(
                            content,
                            filename,
                        )

                        if extracted:
                            documents.extend(extracted)
                            supported_names.append(filename)

                    except Exception:
                        skipped_names.append(filename)

                return (
                    documents,
                    supported_names,
                    skipped_names,
                )

            # Single supported file.
            if is_supported_file(name):

                _, content = download_public_drive_file(
                    drive_id,
                    name,
                )

                documents = extract_document(
                    content,
                    name,
                )

                if documents:
                    supported_names.append(name)

                return (
                    documents,
                    supported_names,
                    skipped_names,
                )

            skipped_names.append(name)

            return (
                documents,
                supported_names,
                skipped_names,
            )

        except PermissionError:
            raise

        except Exception:
            # If metadata lookup fails, fall through to the
            # direct public-file download attempt.
            pass

    # --------------------------------------------------------
    # No API key: try direct public individual-file download.
    # --------------------------------------------------------

    filename_hint = f"drive_file_{drive_id}.pdf"

    try:

        _, content = download_public_drive_file(
            drive_id,
            filename_hint,
        )

    except PermissionError:
        raise

    except Exception as exc:

        if not api_key:
            raise RuntimeError(
                "This link appears to be a Google Drive folder, "
                "or Drive requires API access to enumerate its files. "
                "For folders, add GOOGLE_DRIVE_API_KEY to Streamlit "
                "Secrets. Individual public files do not require "
                "Google OAuth credentials."
            ) from exc

        raise RuntimeError(
            "Unable to access this Google Drive link. "
            "Make sure it is publicly accessible."
        ) from exc

    # Try to determine the actual file type from the response.
    # If no extension is available, inspect common signatures.
    if content.startswith(b"%PDF"):
        filename = filename_hint.replace(
            ".pdf",
            ".pdf",
        )

    elif content[:2] == b"PK":
        # DOCX is a ZIP container.
        filename = filename_hint.replace(
            ".pdf",
            ".docx",
        )

    else:
        filename = filename_hint.replace(
            ".pdf",
            ".txt",
        )

    extracted = extract_document(
        content,
        filename,
    )

    if extracted:
        supported_names.append(filename)
        documents.extend(extracted)

    return (
        documents,
        supported_names,
        skipped_names,
    )


# ============================================================
# DOCUMENT PROCESSING
# ============================================================

def create_document_signature(
    documents: List[Dict],
) -> str:
    """
    Create a stable hash so identical content is not unnecessarily
    re-indexed.
    """

    digest = hashlib.sha256()

    for document in documents:

        text = document["text"]
        metadata = document["metadata"]

        digest.update(
            text.encode(
                "utf-8",
                errors="ignore",
            )
        )

        digest.update(
            str(metadata).encode(
                "utf-8",
                errors="ignore",
            )
        )

    return digest.hexdigest()


def process_documents(
    documents: List[Dict],
    source_name: str,
):
    """
    Complete indexing pipeline:

    extraction -> chunking -> embeddings -> FAISS -> TF-IDF
    """

    if not documents:
        raise ValueError(
            "No readable document content was found."
        )

    chunks = chunk_documents(documents)

    if not chunks:
        raise ValueError(
            "Document extraction succeeded, but no chunks were created."
        )

    signature = create_document_signature(
        documents
    )

    # Avoid rebuilding the exact same index.
    if (
        st.session_state.processed_hash == signature
        and st.session_state.faiss_index is not None
    ):
        return False

    (
        embeddings,
        faiss_index,
        vectorizer,
        tfidf_matrix,
    ) = build_search_index(chunks)

    st.session_state.documents = documents
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.faiss_index = faiss_index
    st.session_state.tfidf_vectorizer = vectorizer
    st.session_state.tfidf_matrix = tfidf_matrix
    st.session_state.processed_hash = signature
    st.session_state.processed_source = source_name

    return True


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def generate_answer(
    question: str,
    retrieved_chunks: List[Dict],
) -> str:
    """
    Send only retrieved document context to Groq.
    """

    api_key = get_groq_api_key()

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets or your environment variables."
        )

    if not retrieved_chunks:
        return (
            "I could not find this information in the uploaded documents."
        )

    context_parts = []

    for number, item in enumerate(
        retrieved_chunks,
        start=1,
    ):

        metadata = item["metadata"]

        filename = metadata.get(
            "filename",
            "Unknown",
        )

        page = metadata.get(
            "page"
        )

        location = filename

        if page:
            location += f", page {page}"

        context_parts.append(
            f"[Source {number}: {location}]\n"
            f"{item['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """
You are a document question-answering assistant.

Answer the user's question ONLY using the supplied document context.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts.
3. If the answer cannot be found in the provided context,
   clearly say that the answer was not found in the uploaded documents.
4. Prefer precise answers.
5. When useful, mention the source document or page.
6. Do not claim that information exists in the documents unless
   it is actually present in the supplied context.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer using ONLY the document context above.
"""

    client = Groq(
        api_key=api_key
    )

    completion = client.chat.completions.create(
        model=get_groq_model(),
        messages=[
            {
                "role": "system",
                "content": system_prompt.strip(),
            },
            {
                "role": "user",
                "content": user_prompt.strip(),
            },
        ],
        temperature=0.1,
    )

    return completion.choices[0].message.content.strip()


# ============================================================
# UI HELPERS
# ============================================================

def render_source(
    source: Dict,
    number: int,
):
    metadata = source["metadata"]

    filename = metadata.get(
        "filename",
        "Unknown file",
    )

    page = metadata.get(
        "page"
    )

    file_type = metadata.get(
        "file_type",
        "",
    )

    if page:
        location = f"Page {page}"
    else:
        location = "Page not available"

    st.markdown(
        f"""
        <div class="source-card">
            <div class="source-title">
                {number}. {filename}
            </div>
            <div class="source-meta">
                {file_type.upper()} · {location}
            </div>
            <div class="source-text">
                {source["text"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_index():
    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.embeddings = None
    st.session_state.faiss_index = None
    st.session_state.tfidf_vectorizer = None
    st.session_state.tfidf_matrix = None
    st.session_state.processed_hash = None
    st.session_state.processed_source = None
    st.session_state.drive_files = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <h1>📚 AI Document Assistant</h1>
        <p>
            A hybrid RAG chatbot for asking questions across your
            PDF, DOCX, TXT, Markdown, and public Google Drive documents.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("### ⚙️ Knowledge Base")

    source_mode = st.radio(
        "Choose document source",
        [
            "Local Files",
            "Google Drive",
        ],
    )

    st.divider()

    st.markdown("### 🔎 Retrieval")

    st.caption(
        "Hybrid retrieval combines semantic vector similarity "
        "with TF-IDF keyword matching."
    )

    st.write(
        f"Semantic weight: **{SEMANTIC_WEIGHT:.0%}**"
    )

    st.write(
        f"Keyword weight: **{KEYWORD_WEIGHT:.0%}**"
    )

    st.divider()

    if st.session_state.faiss_index is not None:

        st.success(
            f"Indexed {len(st.session_state.chunks)} chunks"
        )

        if st.session_state.processed_source:
            st.caption(
                f"Source: {st.session_state.processed_source}"
            )

    else:

        st.info(
            "No documents indexed yet."
        )

    if st.button(
        "Clear Knowledge Base",
        use_container_width=True,
    ):

        clear_index()

        st.success(
            "Knowledge base cleared."
        )

        st.rerun()


# ============================================================
# DOCUMENT SOURCE UI
# ============================================================

if source_mode == "Local Files":

    st.markdown("## 📁 Upload Documents")

    st.markdown(
        """
        <div class="info-card">
            Upload one or more supported documents. The documents
            remain in your current Streamlit session and are indexed
            only when you click <b>Process Documents</b>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose documents",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
            "markdown",
        ],
        accept_multiple_files=True,
        help=(
            "Supported formats: PDF, DOCX, TXT and Markdown."
        ),
    )

    if uploaded_files:

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {len(uploaded_files)}
                    </div>
                    <div class="metric-label">
                        Files selected
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:

            supported_count = sum(
                is_supported_file(
                    file.name
                )
                for file in uploaded_files
            )

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-number">
                        {supported_count}
                    </div>
                    <div class="metric-label">
                        Supported files
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.write("")

        for file in uploaded_files:
            st.caption(
                f"📄 {file.name}"
            )

        if st.button(
            "🚀 Process Documents",
            type="primary",
            use_container_width=True,
        ):

            documents = []

            progress = st.progress(
                0,
                text="Reading documents...",
            )

            try:

                for index, uploaded_file in enumerate(
                    uploaded_files
                ):

                    if not is_supported_file(
                        uploaded_file.name
                    ):
                        st.warning(
                            f"Unsupported file skipped: "
                            f"{uploaded_file.name}"
                        )
                        continue

                    file_bytes = uploaded_file.getvalue()

                    extracted = extract_document(
                        file_bytes,
                        uploaded_file.name,
                    )

                    if not extracted:
                        st.warning(
                            f"No readable text found in "
                            f"{uploaded_file.name}"
                        )
                    else:
                        documents.extend(
                            extracted
                        )

                    progress.progress(
                        (index + 1)
                        / len(uploaded_files),
                        text=(
                            f"Processed "
                            f"{uploaded_file.name}"
                        ),
                    )

                if not documents:
                    st.error(
                        "No readable documents were found."
                    )

                else:

                    with st.spinner(
                        "Creating chunks, embeddings and search indexes..."
                    ):

                        rebuilt = process_documents(
                            documents,
                            "Local Files",
                        )

                    if rebuilt:
                        st.success(
                            f"Successfully indexed "
                            f"{len(documents)} document sections "
                            f"and {len(st.session_state.chunks)} chunks."
                        )
                    else:
                        st.info(
                            "These documents are already indexed."
                        )

            except Exception as exc:

                st.error(
                    f"Document processing failed: {exc}"
                )

            finally:
                progress.empty()


else:

    # ========================================================
    # GOOGLE DRIVE
    # ========================================================

    st.markdown("## ☁️ Google Drive")

    st.markdown(
        """
        <div class="info-card">
            Paste a public Google Drive file or folder link.
            Individual public files can be downloaded without
            Google OAuth. Public folder indexing uses the Google
            Drive API when <code>GOOGLE_DRIVE_API_KEY</code> is
            configured.
        </div>
        """,
        unsafe_allow_html=True,
    )

    drive_url = st.text_input(
        "Google Drive file or folder URL",
        placeholder=(
            "https://drive.google.com/file/d/... "
            "or https://drive.google.com/drive/folders/..."
        ),
    )

    drive_api_key_configured = bool(
        get_drive_api_key()
    )

    if drive_api_key_configured:

        st.caption(
            "✓ Google Drive API access is configured."
        )

    else:

        st.caption(
            "No GOOGLE_DRIVE_API_KEY configured. "
            "Public individual files can still be attempted. "
            "Folder links require the optional Drive API key."
        )

    if st.button(
        "☁️ Load from Drive",
        type="primary",
        use_container_width=True,
    ):

        if not drive_url.strip():

            st.warning(
                "Please paste a Google Drive link first."
            )

        else:

            with st.spinner(
                "Accessing Google Drive..."
            ):

                try:

                    (
                        drive_documents,
                        supported_files,
                        skipped_files,
                    ) = load_drive_source(
                        drive_url
                    )

                    if supported_files:

                        st.success(
                            f"Found {len(supported_files)} "
                            f"supported file(s)."
                        )

                        for filename in supported_files:
                            st.caption(
                                f"✓ {filename}"
                            )

                    if skipped_files:

                        st.warning(
                            "Some files were skipped because "
                            "they are unsupported or could not "
                            "be downloaded."
                        )

                        for filename in skipped_files:
                            st.caption(
                                f"Skipped: {filename}"
                            )

                    if not drive_documents:

                        st.error(
                            "No supported readable documents "
                            "were found at this Drive link."
                        )

                    else:

                        with st.spinner(
                            "Creating chunks, embeddings and search indexes..."
                        ):

                            rebuilt = process_documents(
                                drive_documents,
                                "Google Drive",
                            )

                        if rebuilt:

                            st.success(
                                f"Successfully indexed "
                                f"{len(drive_documents)} document sections "
                                f"into "
                                f"{len(st.session_state.chunks)} chunks."
                            )

                        else:

                            st.info(
                                "These Drive documents are already indexed."
                            )

                except PermissionError as exc:

                    st.error(
                        f"Google Drive access denied: {exc}"
                    )

                except requests.RequestException as exc:

                    st.error(
                        "Could not connect to Google Drive. "
                        f"Network error: {exc}"
                    )

                except Exception as exc:

                    st.error(
                        f"Could not load the Google Drive content: {exc}"
                    )


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

if st.session_state.faiss_index is not None:

    st.divider()

    st.markdown("## 📊 Knowledge Base")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {len(st.session_state.documents)}
                </div>
                <div class="metric-label">
                    Document sections
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-number">
                    {len(st.session_state.chunks)}
                </div>
                <div class="metric-label">
                    Indexed chunks
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-number">
                    Hybrid
                </div>
                <div class="metric-label">
                    Retrieval method
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CHAT INTERFACE
# ============================================================

st.divider()

st.markdown("## 💬 Ask Your Documents")

if st.session_state.faiss_index is None:

    st.info(
        "Upload local documents or load a public Google Drive "
        "file/folder before asking questions."
    )

else:

    # Render previous messages.
    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

            if (
                message["role"] == "assistant"
                and message.get("sources")
            ):

                with st.expander(
                    "📚 Retrieved Sources"
                ):

                    for number, source in enumerate(
                        message["sources"],
                        start=1,
                    ):

                        render_source(
                            source,
                            number,
                        )

    question = st.chat_input(
        "Ask a question about your documents..."
    )

    if question:

        question = question.strip()

        if not question:

            st.warning(
                "Please enter a question."
            )

        else:

            # Show user question.
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message(
                "user"
            ):

                st.markdown(
                    question
                )

            # Retrieve documents.
            with st.chat_message(
                "assistant"
            ):

                with st.spinner(
                    "Searching your documents..."
                ):

                    try:

                        retrieved = hybrid_search(
                            question,
                            FINAL_TOP_K,
                        )

                    except Exception as exc:

                        st.error(
                            f"Search failed: {exc}"
                        )

                        retrieved = []

                if not retrieved:

                    answer = (
                        "I could not find this information "
                        "in the uploaded documents."
                    )

                    st.markdown(
                        answer
                    )

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": [],
                        }
                    )

                else:

                    try:

                        with st.spinner(
                            "Generating answer..."
                        ):

                            answer = generate_answer(
                                question,
                                retrieved,
                            )

                        st.markdown(
                            answer
                        )

                        with st.expander(
                            "📚 Retrieved Sources"
                        ):

                            for number, source in enumerate(
                                retrieved,
                                start=1,
                            ):

                                render_source(
                                    source,
                                    number,
                                )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": retrieved,
                            }
                        )

                    except Exception as exc:

                        st.error(
                            f"Could not generate the answer: {exc}"
                        )

                        st.session_state.chat_history.append(
                            {
                                "role": "assistant",
                                "content": (
                                    "I couldn't generate an answer "
                                    "because the LLM request failed."
                                ),
                                "sources": retrieved,
                            }
                        )
