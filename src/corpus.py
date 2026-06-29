"""
corpus.py
Dual corpus loader for Week 3 AVS.

Layer 1 — Global governance frameworks (ARM™, AVRE™, EU AI Act, NIST AI RMF, ISO 42001)
           Lives in corpus/layer1/
Layer 2 — Client-specific (Meridian: company policy, sector regulation, cost parameters)
           Lives in corpus/layer2/

Each layer gets its own TFIDFVectorStore so retrieval can be scoped per intent:
  - SCORE / CLARIFY / BRIEF  → both layers
  - COMPLIANCE               → layer2 (regulation-heavy)
  - SEARCH                   → both layers, layer tag in metadata
"""

import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CORPUS_DIR   = Path(__file__).parent.parent / "corpus"
L1_DIR       = CORPUS_DIR / "layer1"
L2_DIR       = CORPUS_DIR / "layer2"
STORE_PATH   = Path(__file__).parent.parent / "dual_vectorstore.pkl"


# ── Vectorstore (same TF-IDF class as Week 2, extended with layer metadata) ──

class TFIDFVectorStore:
    """TF-IDF vectorstore with MMR diversity retrieval and layer-aware metadata."""

    def __init__(self, layer_name: str = "global"):
        self.layer_name = layer_name
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        self.chunks: List[Document] = []
        self.matrix = None

    def add_documents(self, docs: List[Document]):
        self.chunks = docs
        texts = [d.page_content for d in docs]
        self.matrix = self.vectorizer.fit_transform(texts)
        print(f"[Corpus:{self.layer_name}] {len(docs)} chunks · {self.matrix.shape[1]} TF-IDF features")

    def similarity_search_with_mmr(
        self,
        query: str,
        k: int = 5,
        fetch_k: int = 20,
        lambda_mult: float = 0.6,
    ) -> List[Document]:
        if self.matrix is None or len(self.chunks) == 0:
            return []
        query_vec = self.vectorizer.transform([query])
        scores    = cosine_similarity(query_vec, self.matrix).flatten()
        top_idx   = list(np.argsort(scores)[::-1][:fetch_k])

        selected, remaining = [], top_idx[:]
        while len(selected) < k and remaining:
            if not selected:
                best = remaining[0]
            else:
                sel_vecs = self.matrix[selected]
                mmr_scores = []
                for idx in remaining:
                    rel  = scores[idx]
                    sim  = float(cosine_similarity(self.matrix[idx], sel_vecs).max())
                    mmr_scores.append((idx, lambda_mult * rel - (1 - lambda_mult) * sim))
                best = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best)
            remaining.remove(best)

        return [self.chunks[i] for i in selected]


# ── Corpus loading helpers ────────────────────────────────────────────────────

def _load_txt_files(directory: Path, layer_tag: str) -> List[Document]:
    """Load all .txt files from a directory, chunk, and tag with layer metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks: List[Document] = []
    if not directory.exists():
        print(f"[Corpus] Warning: {directory} not found — skipping {layer_tag}")
        return all_chunks

    for txt_file in sorted(directory.glob("**/*.txt")):
        text   = txt_file.read_text(encoding="utf-8")
        doc    = Document(page_content=text, metadata={
            "source_file": txt_file.stem,
            "layer": layer_tag,
        })
        chunks = splitter.split_documents([doc])
        for c in chunks:
            c.metadata["source_file"] = txt_file.stem
            c.metadata["layer"]       = layer_tag
        all_chunks.extend(chunks)
        print(f"[Corpus:{layer_tag}]   {txt_file.stem}: {len(chunks)} chunks")

    print(f"[Corpus:{layer_tag}] Total: {len(all_chunks)} chunks")
    return all_chunks


# ── Dual store builder ────────────────────────────────────────────────────────

class DualCorpus:
    """
    Holds two TFIDFVectorStore instances.
    Provides unified retrieval that can target layer1, layer2, or both.
    """

    def __init__(self, store_l1: TFIDFVectorStore, store_l2: TFIDFVectorStore):
        self.l1 = store_l1
        self.l2 = store_l2

    def retrieve(
        self,
        query: str,
        layers: str = "both",   # "l1" | "l2" | "both"
        k: int = 5,
    ) -> Dict:
        """
        Retrieve context from the specified layer(s).
        Returns dict with keys: context, sources, chunks, layer_breakdown.
        """
        docs: List[Document] = []

        if layers in ("l1", "both"):
            docs += self.l1.similarity_search_with_mmr(query, k=k)
        if layers in ("l2", "both"):
            docs += self.l2.similarity_search_with_mmr(query, k=k)

        # Deduplicate by content hash
        seen, unique = set(), []
        for d in docs:
            h = hash(d.page_content[:100])
            if h not in seen:
                seen.add(h)
                unique.append(d)

        context = "\n\n---\n\n".join(d.page_content for d in unique)
        sources = list({d.metadata.get("source_file", "unknown") for d in unique})
        layer_breakdown = {
            "l1_chunks": [d.metadata.get("source_file") for d in unique if d.metadata.get("layer") == "l1"],
            "l2_chunks": [d.metadata.get("source_file") for d in unique if d.metadata.get("layer") == "l2"],
        }

        return {
            "context":         context,
            "sources":         sources,
            "num_chunks":      len(unique),
            "layer_breakdown": layer_breakdown,
        }

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Corpus] Dual store saved → {path}")

    @staticmethod
    def load(path: Path) -> "DualCorpus":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[Corpus] Dual store loaded (L1:{len(obj.l1.chunks)} · L2:{len(obj.l2.chunks)} chunks)")
        return obj


def build_dual_corpus(force_rebuild: bool = False) -> DualCorpus:
    """Build or load the dual TF-IDF corpus."""
    if STORE_PATH.exists() and not force_rebuild:
        return DualCorpus.load(STORE_PATH)

    print("[Corpus] Building dual corpus from scratch...")

    # Layer 1 — global frameworks
    l1_chunks = _load_txt_files(L1_DIR, "l1")
    store_l1  = TFIDFVectorStore(layer_name="l1")
    if l1_chunks:
        store_l1.add_documents(l1_chunks)

    # Layer 2 — client-specific (Meridian)
    l2_chunks = _load_txt_files(L2_DIR, "l2")
    store_l2  = TFIDFVectorStore(layer_name="l2")
    if l2_chunks:
        store_l2.add_documents(l2_chunks)

    dual = DualCorpus(store_l1, store_l2)
    dual.save(STORE_PATH)
    return dual
