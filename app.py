import os
import re
from io import BytesIO
from typing import Any, Dict, List, Tuple

import numpy as np
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

APP_TITLE = "📚 RAG Document Q&A Assistant"

# Lightweight local embedding model.
# It runs locally and does NOT send documents to Groq.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Current Groq production model recommended as a replacement
# for deprecated older models.
GROQ_MODEL = "openai/gpt-oss-120b"

# Chunking configuration.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# Retrieval configuration.
SEMANTIC_TOP_K = 8
KEYWORD_TOP_K = 8
FINAL_TOP_K = 6

# Weight used by the hybrid retriever.
SEMANTIC_WEIGHT = 0.65
KEYWORD_WEIGHT = 0.35


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #666;
            margin-bottom: 1.5rem;
        }

        .source-box {
            padding: 0.8rem;
            border-radius: 0.5rem;
            border: 1px solid #ddd;
            margin-bottom: 0.7rem;
        }

        .source-title {
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def initialize_session_state():
    """Initialize all application state variables."""

    defaults = {
        "documents": [],
        "chunks": [],
        "embeddings": None,
        "tfidf_vectorizer": None,
        "tfidf_matrix": None,
        "processed": False,
        "chat_history": [],
        "file_signature": None,
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
    """
    Load the local Sentence Transformer model once.

    Streamlit caches this resource so the embedding model is
    not downloaded/loaded again on every interaction.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ============================================================
# GROQ API KEY
# ============================================================

def get_groq_api_key() -> str:
    """
    Get GROQ_API_KEY from:

    1. Environment variable
    2. Streamlit secrets

    The key is never hardcoded.
    """

    # First check environment variables.
    api_key = os.getenv("GROQ_API_KEY")

    if api_key:
        return api_key.strip()

    # Then check Streamlit secrets.
    try:
        if "GROQ_API_KEY" in st.secrets:
            return str(st.secrets["GROQ_API_KEY"]).strip()
    except Exception:
        pass

    return ""


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """Clean extracted document text."""

    if not text:
        return ""

    # Normalize different newline styles.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove excessive whitespace while preserving paragraphs.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_text(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract PDF text page-by-page.

    Page numbers are preserved in metadata.
    """

    pages = []

    try:
        reader = PdfReader(BytesIO(file_bytes))

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
                text = clean_text(text)

                if text:
                    pages.append(
                        {
                            "text": text,
                            "metadata": {
                                "filename": filename,
                                "file_type": "pdf",
                                "page": page_number,
                            },
                        }
                    )

            except Exception as page_error:
                st.warning(
                    f"Could not extract page {page_number} "
                    f"from {filename}: {page_error}"
                )

    except Exception as error:
        raise ValueError(f"Failed to read PDF '{filename}': {error}")

    return pages


# ============================================================
# DOCX EXTRACTION
# ============================================================

def extract_docx_text(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract text from a DOCX document.

    DOCX files do not have a reliable page structure at this
    extraction level, so filename is preserved as metadata.
    """

    try:
        document = Document(BytesIO(file_bytes))

        paragraphs = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        text = "\n\n".join(paragraphs)
        text = clean_text(text)

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

    except Exception as error:
        raise ValueError(f"Failed to read DOCX '{filename}': {error}")


# ============================================================
# TXT / MARKDOWN EXTRACTION
# ============================================================

def extract_text_file(
    file_bytes: bytes,
    filename: str,
    file_type: str,
) -> List[Dict[str, Any]]:
    """
    Extract TXT or Markdown files.

    Filename and file type are preserved as metadata.
    """

    try:
        # UTF-8 first.
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback for files containing non-UTF8 characters.
            text = file_bytes.decode("latin-1")

        text = clean_text(text)

        if not text:
            return []

        return [
            {
                "text": text,
                "metadata": {
                    "filename": filename,
                    "file_type": file_type,
                    "page": None,
                },
            }
        ]

    except Exception as error:
        raise ValueError(f"Failed to read '{filename}': {error}")


# ============================================================
# GENERIC DOCUMENT EXTRACTION
# ============================================================

def extract_document(file) -> List[Dict[str, Any]]:
    """
    Detect the uploaded file type and call the appropriate
    extraction function.
    """

    filename = file.name
    extension = os.path.splitext(filename)[1].lower()

    file_bytes = file.getvalue()

    if extension == ".pdf":
        return extract_pdf_text(file_bytes, filename)

    elif extension == ".docx":
        return extract_docx_text(file_bytes, filename)

    elif extension == ".txt":
        return extract_text_file(file_bytes, filename, "txt")

    elif extension in [".md", ".markdown"]:
        return extract_text_file(file_bytes, filename, "markdown")

    else:
        raise ValueError(
            f"Unsupported file type: {extension or 'unknown'}"
        )


# ============================================================
# CHUNKING
# ============================================================

def split_text_into_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping word-based chunks.

    Character-based size is used indirectly by converting
    approximately to words. This keeps implementation simple
    while providing overlap between chunks.
    """

    if not text:
        return []

    # Approximate chunk sizes using words.
    words = text.split()

    # Approximately 4-5 characters per word is common for English.
    words_per_chunk = max(50, chunk_size // 5)
    overlap_words = max(10, overlap // 5)

    if overlap_words >= words_per_chunk:
        overlap_words = words_per_chunk // 3

    chunks = []

    start = 0

    while start < len(words):
        end = min(start + words_per_chunk, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - overlap_words

    return chunks


def chunk_documents(extracted_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create overlapping chunks while preserving document metadata.
    """

    all_chunks = []

    for document in extracted_documents:
        text = document["text"]
        metadata = document["metadata"]

        text_chunks = split_text_into_chunks(text)

        for chunk_index, chunk in enumerate(text_chunks):
            chunk_metadata = metadata.copy()

            chunk_metadata["chunk_id"] = chunk_index

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
    chunks: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Generate local semantic embeddings.

    Groq is NOT used here.
    """

    if not chunks:
        raise ValueError("No chunks available for embedding.")

    model = load_embedding_model()

    texts = [chunk["text"] for chunk in chunks]

    # normalize_embeddings=True makes dot product equivalent
    # to cosine similarity.
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return embeddings.astype("float32")


# ============================================================
# SEARCH INDEX
# ============================================================

def build_search_index(
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
) -> Tuple[TfidfVectorizer, Any]:
    """
    Build the keyword search index using TF-IDF.

    Semantic/vector search uses the embeddings separately.
    """

    if not chunks:
        raise ValueError("Cannot build an index without chunks.")

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=50000,
        sublinear_tf=True,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)

    return vectorizer, tfidf_matrix


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    top_k: int = SEMANTIC_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Semantic search:

    Question -> embedding -> cosine similarity against
    document chunk embeddings.
    """

    if not chunks or embeddings is None:
        return []

    model = load_embedding_model()

    # Create an embedding for the user's question.
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].astype("float32")

    # Because both vectors are normalized, dot product = cosine similarity.
    scores = embeddings @ query_embedding

    top_k = min(top_k, len(chunks))

    # Get the highest scoring chunks.
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        result = {
            "index": int(index),
            "score": float(scores[index]),
            "text": chunks[index]["text"],
            "metadata": chunks[index]["metadata"],
            "search_type": "semantic",
        }

        results.append(result)

    return results


# ============================================================
# KEYWORD SEARCH
# ============================================================

def keyword_search(
    query: str,
    chunks: List[Dict[str, Any]],
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_k: int = KEYWORD_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Keyword search using TF-IDF.

    This finds chunks containing terms that are important
    to the user's query.
    """

    if (
        not chunks
        or vectorizer is None
        or tfidf_matrix is None
    ):
        return []

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_matrix,
    ).flatten()

    top_k = min(top_k, len(chunks))

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        # Don't include completely unrelated chunks.
        if scores[index] <= 0:
            continue

        result = {
            "index": int(index),
            "score": float(scores[index]),
            "text": chunks[index]["text"],
            "metadata": chunks[index]["metadata"],
            "search_type": "keyword",
        }

        results.append(result)

    return results


# ============================================================
# HYBRID SEARCH
# ============================================================

def normalize_scores(results: List[Dict[str, Any]]) -> Dict[int, float]:
    """Normalize scores to the 0-1 range."""

    if not results:
        return {}

    scores = [item["score"] for item in results]

    minimum = min(scores)
    maximum = max(scores)

    if maximum == minimum:
        return {
            item["index"]: 1.0
            for item in results
        }

    return {
        item["index"]: (item["score"] - minimum) / (maximum - minimum)
        for item in results
    }


def hybrid_search(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    top_k: int = FINAL_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Combine semantic and keyword retrieval.

    Hybrid score:

        65% semantic similarity
        +
        35% keyword similarity
    """

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    semantic_results = semantic_search(
        query=query,
        chunks=chunks,
        embeddings=embeddings,
        top_k=SEMANTIC_TOP_K,
    )

    # --------------------------------------------------------
    # Keyword retrieval
    # --------------------------------------------------------

    keyword_results = keyword_search(
        query=query,
        chunks=chunks,
        vectorizer=vectorizer,
        tfidf_matrix=tfidf_matrix,
        top_k=KEYWORD_TOP_K,
    )

    # Normalize both score types separately because their
    # numerical ranges can be different.
    semantic_scores = normalize_scores(semantic_results)
    keyword_scores = normalize_scores(keyword_results)

    # Store all candidate chunk IDs.
    candidate_indices = set(
        semantic_scores.keys()
    ).union(
        keyword_scores.keys()
    )

    combined_results = []

    for index in candidate_indices:

        semantic_score = semantic_scores.get(index, 0.0)
        keyword_score = keyword_scores.get(index, 0.0)

        # ----------------------------------------------------
        # Hybrid scoring
        # ----------------------------------------------------

        hybrid_score = (
            SEMANTIC_WEIGHT * semantic_score
            + KEYWORD_WEIGHT * keyword_score
        )

        combined_results.append(
            {
                "index": index,
                "semantic_score": semantic_score,
                "keyword_score": keyword_score,
                "hybrid_score": hybrid_score,
                "text": chunks[index]["text"],
                "metadata": chunks[index]["metadata"],
            }
        )

    # Sort by final hybrid score.
    combined_results.sort(
        key=lambda x: x["hybrid_score"],
        reverse=True,
    )

    # Return only the strongest results.
    return combined_results[:top_k]


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(
    retrieved_chunks: List[Dict[str, Any]]
) -> str:
    """
    Convert retrieved chunks into a structured context that
    will be sent to the LLM.
    """

    context_parts = []

    for number, result in enumerate(retrieved_chunks, start=1):

        metadata = result["metadata"]

        filename = metadata.get(
            "filename",
            "Unknown file",
        )

        page = metadata.get("page")

        if page:
            source = f"{filename}, page {page}"
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

    return "\n".join(context_parts)


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def generate_answer(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Send ONLY retrieved document context and the user's
    question to Groq.

    The system prompt explicitly prevents the model from
    using outside knowledge.
    """

    api_key = get_groq_api_key()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets or your environment."
        )

    if not retrieved_chunks:
        return (
            "The answer was not found in the uploaded documents."
        )

    client = Groq(api_key=api_key)

    context = build_context(retrieved_chunks)

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

    # --------------------------------------------------------
    # Retrieved context is sent to Groq here.
    # --------------------------------------------------------

    completion = client.chat.completions.create(
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

    # --------------------------------------------------------
    # Final answer generated by the Groq LLM.
    # --------------------------------------------------------

    answer = completion.choices[0].message.content

    if not answer:
        raise ValueError(
            "Groq returned an empty response."
        )

    return answer.strip()


# ============================================================
# FILE SIGNATURE
# ============================================================

def get_file_signature(files) -> Tuple:
    """
    Create a lightweight signature of uploaded files.

    This helps detect when the user has changed the files
    after processing.
    """

    return tuple(
        (
            file.name,
            file.size,
            getattr(file, "type", None),
        )
        for file in files
    )


# ============================================================
# RESET FUNCTION
# ============================================================

def reset_application():
    """Reset processed documents and chat history."""

    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.embeddings = None
    st.session_state.tfidf_vectorizer = None
    st.session_state.tfidf_matrix = None
    st.session_state.processed = False
    st.session_state.chat_history = []
    st.session_state.file_signature = None


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">📚 RAG Document Q&A Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Upload PDF, DOCX, TXT or Markdown documents and ask questions
    using a hybrid RAG pipeline combining semantic search,
    keyword search and Groq-powered answer generation.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ RAG Configuration")

    st.write(
        f"**Embedding model:** `{EMBEDDING_MODEL_NAME}`"
    )

    st.write(
        f"**LLM:** `{GROQ_MODEL}`"
    )

    st.write(
        f"**Semantic weight:** `{SEMANTIC_WEIGHT:.0%}`"
    )

    st.write(
        f"**Keyword weight:** `{KEYWORD_WEIGHT:.0%}`"
    )

    st.write(
        f"**Final chunks:** `{FINAL_TOP_K}`"
    )

    st.divider()

    if st.session_state.processed:

        st.success("Documents processed")

        st.metric(
            "Indexed chunks",
            len(st.session_state.chunks),
        )

    else:
        st.info("No document index created yet.")

    if st.button(
        "🗑️ Reset Application",
        use_container_width=True,
    ):
        reset_application()
        st.rerun()


# ============================================================
# DOCUMENT UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "📄 Upload documents",
    type=[
        "pdf",
        "docx",
        "txt",
        "md",
        "markdown",
    ],
    accept_multiple_files=True,
    help=(
        "You can upload multiple PDF, DOCX, TXT or Markdown "
        "files at the same time."
    ),
)


# ============================================================
# UPLOAD STATE DETECTION
# ============================================================

if uploaded_files:

    current_signature = get_file_signature(uploaded_files)

    # If the uploaded files changed after the previous processing,
    # mark the current index as stale.
    if (
        st.session_state.file_signature is not None
        and current_signature != st.session_state.file_signature
    ):
        st.session_state.processed = False

        st.info(
            "The uploaded files changed. "
            "Click 'Process Documents' to rebuild the index."
        )


# ============================================================
# PROCESS DOCUMENTS BUTTON
# ============================================================

process_button = st.button(
    "🚀 Process Documents",
    type="primary",
    use_container_width=True,
)


if process_button:

    # --------------------------------------------------------
    # Validate upload
    # --------------------------------------------------------

    if not uploaded_files:

        st.warning(
            "Please upload at least one document before processing."
        )

    else:

        with st.status(
            "Processing documents...",
            expanded=True,
        ) as status:

            try:

                # ------------------------------------------------
                # STEP 1: Extract text
                # ------------------------------------------------

                st.write("📖 Extracting document text...")

                extracted_documents = []

                for uploaded_file in uploaded_files:

                    extension = os.path.splitext(
                        uploaded_file.name
                    )[1].lower()

                    supported_extensions = {
                        ".pdf",
                        ".docx",
                        ".txt",
                        ".md",
                        ".markdown",
                    }

                    if extension not in supported_extensions:
                        st.warning(
                            f"Skipping unsupported file: "
                            f"{uploaded_file.name}"
                        )
                        continue

                    try:

                        documents = extract_document(
                            uploaded_file
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
                            f"'{uploaded_file.name}': {error}"
                        )

                if not extracted_documents:
                    raise ValueError(
                        "No readable text was extracted "
                        "from the uploaded documents."
                    )

                st.write(
                    f"✅ Extracted text from "
                    f"{len(extracted_documents)} document sections."
                )

                # ------------------------------------------------
                # STEP 2: Chunk documents
                # ------------------------------------------------

                st.write("✂️ Splitting documents into chunks...")

                chunks = chunk_documents(
                    extracted_documents
                )

                if not chunks:
                    raise ValueError(
                        "No chunks were created from the documents."
                    )

                st.write(
                    f"✅ Created {len(chunks)} text chunks."
                )

                # ------------------------------------------------
                # STEP 3: Create embeddings
                # ------------------------------------------------

                st.write(
                    "🧠 Creating local document embeddings..."
                )

                embeddings = create_embeddings(chunks)

                st.write(
                    f"✅ Created embeddings with "
                    f"{embeddings.shape[1]} dimensions."
                )

                # ------------------------------------------------
                # STEP 4: Build keyword index
                # ------------------------------------------------

                st.write(
                    "🔎 Building keyword search index..."
                )

                vectorizer, tfidf_matrix = build_search_index(
                    chunks,
                    embeddings,
                )

                st.write(
                    "✅ Keyword and semantic indexes ready."
                )

                # ------------------------------------------------
                # STEP 5: Store everything in session state
                # ------------------------------------------------

                st.session_state.documents = extracted_documents

                st.session_state.chunks = chunks

                st.session_state.embeddings = embeddings

                st.session_state.tfidf_vectorizer = vectorizer

                st.session_state.tfidf_matrix = tfidf_matrix

                st.session_state.processed = True

                st.session_state.file_signature = (
                    get_file_signature(uploaded_files)
                )

                # Start a fresh chat when a new index is created.
                st.session_state.chat_history = []

                status.update(
                    label="✅ Documents processed successfully!",
                    state="complete",
                    expanded=False,
                )

                st.success(
                    f"Successfully indexed "
                    f"{len(chunks)} chunks."
                )

            except Exception as error:

                status.update(
                    label="❌ Document processing failed",
                    state="error",
                    expanded=True,
                )

                st.error(
                    f"Processing failed: {error}"
                )


# ============================================================
# INDEX STATUS
# ============================================================

if st.session_state.processed:

    st.success(
        f"📚 Knowledge base ready — "
        f"{len(st.session_state.chunks)} chunks indexed."
    )


# ============================================================
# CHAT SECTION
# ============================================================

st.divider()

st.subheader("💬 Ask Questions")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "🔎 Retrieved Sources"
            ):

                for index, source in enumerate(
                    message["sources"],
                    start=1,
                ):

                    metadata = source["metadata"]

                    filename = metadata.get(
                        "filename",
                        "Unknown file",
                    )

                    page = metadata.get("page")

                    if page:
                        location = (
                            f"{filename} — "
                            f"Page {page}"
                        )
                    else:
                        location = filename

                    st.markdown(
                        f"**Source {index}: {location}**"
                    )

                    # Display retrieval scores for teaching/demo.
                    score_text = (
                        f"Hybrid: {source['hybrid_score']:.3f} | "
                        f"Semantic: {source['semantic_score']:.3f} | "
                        f"Keyword: {source['keyword_score']:.3f}"
                    )

                    st.caption(score_text)

                    st.code(
                        source["text"],
                        language="text",
                    )

                    if index < len(message["sources"]):
                        st.divider()


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your uploaded documents..."
)


if question:

    question = question.strip()

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:

        st.warning(
            "Please enter a question."
        )

    elif not st.session_state.processed:

        st.warning(
            "Please upload and process documents before "
            "asking a question."
        )

    else:

        # ----------------------------------------------------
        # Display user message
        # ----------------------------------------------------

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # ----------------------------------------------------
        # Retrieve documents
        # ----------------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching documents and generating answer..."
            ):

                try:

                    # ------------------------------------------------
                    # Hybrid retrieval happens here.
                    # ------------------------------------------------

                    retrieved_chunks = hybrid_search(
                        query=question,
                        chunks=st.session_state.chunks,
                        embeddings=st.session_state.embeddings,
                        vectorizer=st.session_state.tfidf_vectorizer,
                        tfidf_matrix=st.session_state.tfidf_matrix,
                        top_k=FINAL_TOP_K,
                    )

                    if not retrieved_chunks:

                        answer = (
                            "The answer was not found in "
                            "the uploaded documents."
                        )

                    else:

                        # ------------------------------------------------
                        # Send retrieved context to Groq.
                        # ------------------------------------------------

                        answer = generate_answer(
                            question=question,
                            retrieved_chunks=retrieved_chunks,
                        )

                    # ------------------------------------------------
                    # Display final answer.
                    # ------------------------------------------------

                    st.markdown(answer)

                    # ------------------------------------------------
                    # Display retrieved sources.
                    # ------------------------------------------------

                    if retrieved_chunks:

                        with st.expander(
                            "🔎 Retrieved Sources"
                        ):

                            for index, source in enumerate(
                                retrieved_chunks,
                                start=1,
                            ):

                                metadata = source["metadata"]

                                filename = metadata.get(
                                    "filename",
                                    "Unknown file",
                                )

                                page = metadata.get(
                                    "page"
                                )

                                if page:
                                    location = (
                                        f"{filename} — "
                                        f"Page {page}"
                                    )
                                else:
                                    location = filename

                                st.markdown(
                                    f"**Source {index}: "
                                    f"{location}**"
                                )

                                st.caption(
                                    f"Hybrid: "
                                    f"{source['hybrid_score']:.3f} | "
                                    f"Semantic: "
                                    f"{source['semantic_score']:.3f} | "
                                    f"Keyword: "
                                    f"{source['keyword_score']:.3f}"
                                )

                                st.code(
                                    source["text"],
                                    language="text",
                                )

                                if index < len(
                                    retrieved_chunks
                                ):
                                    st.divider()

                    # ------------------------------------------------
                    # Save assistant response to chat history.
                    # ------------------------------------------------

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": retrieved_chunks,
                        }
                    )

                except Exception as error:

                    error_message = (
                        f"Sorry, an error occurred: {error}"
                    )

                    st.error(error_message)

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

st.divider()

st.caption(
    "RAG Pipeline: "
    "Upload → Extract → Chunk → Embed → Index → "
    "Hybrid Search → Context → Groq → Answer"
)
