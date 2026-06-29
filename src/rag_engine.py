"""
rag_engine.py
RAG pipeline for the Agentification Value Scorer.
Uses TF-IDF + cosine similarity (sklearn) — no external model downloads.
Hybrid retrieval: TF-IDF similarity + MMR-style diversity selection.
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
STORE_PATH = Path(__file__).parent.parent / "vectorstore.pkl"


class TFIDFVectorStore:
    """
    Lightweight TF-IDF vectorstore with MMR diversity retrieval.
    No API calls, no model downloads. Fully offline.
    """
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=8000,
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
        print(f"[RAG] Vectorstore built: {len(docs)} chunks, {self.matrix.shape[1]} TF-IDF features")

    def similarity_search_with_mmr(self, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.6) -> List[Document]:
        """MMR: balance relevance and diversity among retrieved chunks."""
        if self.matrix is None:
            return []
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()
        top_indices = list(np.argsort(scores)[::-1][:fetch_k])

        selected = []
        remaining = top_indices[:]
        while len(selected) < k and remaining:
            if not selected:
                best = remaining[0]
            else:
                selected_vecs = self.matrix[selected]
                mmr_scores = []
                for idx in remaining:
                    relevance = scores[idx]
                    sim_to_selected = float(cosine_similarity(self.matrix[idx], selected_vecs).max())
                    mmr_scores.append((idx, lambda_mult * relevance - (1 - lambda_mult) * sim_to_selected))
                best = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best)
            remaining.remove(best)

        return [self.chunks[i] for i in selected]

    def save(self, path: Path):
        path.parent.mkdir(exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[RAG] Vectorstore saved to {path}")

    @staticmethod
    def load(path: Path) -> "TFIDFVectorStore":
        with open(path, "rb") as f:
            store = pickle.load(f)
        print(f"[RAG] Loaded vectorstore ({len(store.chunks)} chunks)")
        return store


def load_and_chunk_corpus() -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for txt_file in sorted(CORPUS_DIR.glob("**/*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        doc = Document(page_content=text, metadata={"source_file": txt_file.stem})
        file_chunks = splitter.split_documents([doc])
        for c in file_chunks:
            c.metadata["source_file"] = txt_file.stem
        chunks.extend(file_chunks)
        print(f"[RAG]   {txt_file.stem}: {len(file_chunks)} chunks")
    print(f"[RAG] Total: {len(chunks)} chunks")
    return chunks


def build_vectorstore(force_rebuild: bool = False) -> TFIDFVectorStore:
    if STORE_PATH.exists() and not force_rebuild:
        return TFIDFVectorStore.load(STORE_PATH)
    print("[RAG] Building vectorstore from corpus...")
    chunks = load_and_chunk_corpus()
    store = TFIDFVectorStore()
    store.add_documents(chunks)
    store.save(STORE_PATH)
    return store


def get_retriever(store: TFIDFVectorStore, k: int = 5):
    class Retriever:
        def __init__(self, s, k):
            self.store = s
            self.k = k
        def invoke(self, query: str) -> List[Document]:
            return self.store.similarity_search_with_mmr(query, k=self.k)
    return Retriever(store, k)


def retrieve_context(query: str, retriever) -> Dict[str, Any]:
    docs = retriever.invoke(query)
    context_text = "\n\n---\n\n".join(d.page_content for d in docs)
    sources = list({d.metadata.get("source_file", "unknown") for d in docs})
    return {"context": context_text, "sources": sources, "num_chunks": len(docs)}


if __name__ == "__main__":
    store = build_vectorstore(force_rebuild=True)
    retriever = get_retriever(store)
    result = retrieve_context("What is the ARM autonomy risk score for an HR hiring agent?", retriever)
    print(f"\nSources: {result['sources']}")
    print(f"Chunks: {result['num_chunks']}")
    print(f"Preview:\n{result['context'][:500]}")
