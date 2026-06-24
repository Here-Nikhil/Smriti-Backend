"""
rag_engine.py
--------------
This is the brain of the chatbot. It has 4 jobs:
  1. Extract text from PDFs, keeping track of WHICH PAGE each bit of text came from
  2. Split that text into small overlapping chunks
  3. Turn chunks into embeddings (vectors) and store them in a FAISS index
  4. Given a question, find the most relevant chunks and return them with citations
"""

import os
import pickle
import numpy as np
import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# This model runs locally on CPU, no API key needed, ~80MB download.
# It converts text into a 384-dimensional vector that captures meaning.
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

CHUNK_SIZE = 800       # characters per chunk (roughly ~150-200 words)
CHUNK_OVERLAP = 150    # overlap so we don't cut a sentence/idea in half between chunks


class Chunk:
    """A single chunk of text plus the metadata we need for citations."""
    def __init__(self, text, source, page):
        self.text = text
        self.source = source   # original filename, e.g. "handbook.pdf"
        self.page = page       # 1-indexed page number

    def to_dict(self):
        return {"text": self.text, "source": self.source, "page": self.page}

    @staticmethod
    def from_dict(d):
        return Chunk(d["text"], d["source"], d["page"])


def extract_pages(pdf_path, filename):
    """
    Read a PDF and return a list of (page_number, page_text) tuples.
    Keeping page numbers here is what lets us cite "page 4" later instead
    of just dumping one giant blob of text per PDF.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((i + 1, text))
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Simple sliding-window chunker. We chunk by characters (not tokens) because
    it's dependency-free and good enough for this project. Overlap means the
    end of chunk N repeats at the start of chunk N+1, so an idea that spans
    the chunk boundary isn't lost entirely in either chunk.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # step forward, but re-include the overlap
    return chunks


def build_chunks_from_pdfs(pdf_paths_and_names):
    """
    pdf_paths_and_names: list of (filepath, original_filename)
    Returns a flat list of Chunk objects across all PDFs, each tagged with
    its source filename and page number.
    """
    all_chunks = []
    for path, filename in pdf_paths_and_names:
        pages = extract_pages(path, filename)
        for page_num, page_text in pages:
            for piece in chunk_text(page_text):
                all_chunks.append(Chunk(piece, filename, page_num))
    return all_chunks


class VectorStore:
    """
    Wraps a FAISS index + the chunk metadata that goes with each vector.
    FAISS only stores numbers (vectors) -- it has no idea what text or page
    each vector came from. So we keep a parallel Python list (self.chunks)
    where chunks[i] is the metadata for the vector at index i in FAISS.
    """

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
        self.index = None
        self.chunks = []  # list[Chunk], same order as vectors in self.index

    def build(self, chunks):
        """Embed all chunks and build a fresh FAISS index from scratch."""
        self.chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        # IndexFlatIP = exact search using inner product. Since we normalized
        # the vectors above, inner product is equivalent to cosine similarity.
        # "Flat" means no approximation -- perfect accuracy, fine at this scale
        # (thousands of chunks). At millions of chunks you'd switch to an
        # approximate index like IVF or HNSW for speed.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

    def add(self, new_chunks):
        """Add more chunks to an existing index (e.g. user uploads another PDF)."""
        if self.index is None:
            self.build(new_chunks)
            return
        texts = [c.text for c in new_chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.chunks.extend(new_chunks)

    def search(self, query, k=4):
        """
        Embed the query, find the k most similar chunks.
        Returns list of (Chunk, similarity_score).
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        query_vec = self.model.encode([query], normalize_embeddings=True)
        query_vec = np.array(query_vec, dtype="float32")
        scores, indices = self.index.search(query_vec, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, dir_path):
        """Persist the index + metadata to disk so we don't re-embed every restart."""
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "chunks.pkl"), "wb") as f:
            pickle.dump([c.to_dict() for c in self.chunks], f)

    def load(self, dir_path):
        self.index = faiss.read_index(os.path.join(dir_path, "index.faiss"))
        with open(os.path.join(dir_path, "chunks.pkl"), "rb") as f:
            raw = pickle.load(f)
        self.chunks = [Chunk.from_dict(d) for d in raw]


def format_context_with_citations(results):
    """
    Turn search results into a text block for the LLM prompt, AND a separate
    list of citation strings for the UI. Keeping these separate matters:
    the LLM sees labeled sources so it can reference them, and the UI can
    show clean citation chips without re-parsing the LLM's output.
    """
    context_parts = []
    citations = []
    for i, (chunk, score) in enumerate(results):
        tag = f"[Source {i+1}: {chunk.source}, page {chunk.page}]"
        context_parts.append(f"{tag}\n{chunk.text}")
        citations.append({"source": chunk.source, "page": chunk.page, "score": round(score, 3)})
    return "\n\n".join(context_parts), citations
