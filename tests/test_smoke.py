import importlib
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from qdrant.tfidf import TfidfRAG
from scincl.core import SciNCLIngestion

PUBMED_CSV = Path(__file__).parents[1] / "data" / "pubmed_abstracts.csv"


@pytest.mark.parametrize(
    "module",
    [
        "main",
        "qdrant",
        "qdrant.cli",
        "scincl",
        "scincl.cli",
        "eval.benchmark",
        # Transitive deps that Dependabot bumps independently.
        "filelock",
        "milvus_lite",
        "FlagEmbedding",
    ],
)
def test_imports(module):
    importlib.import_module(module)


@pytest.fixture
def tfidf(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rag = TfidfRAG(vectorizer_path=str(tmp_path / "vectorizer.pkl"))
    rag.qdrant_client = QdrantClient(":memory:")
    return rag


def test_tfidf_indexes_and_searches_pubmed(tfidf):
    docs = tfidf.process_pubmed_data(str(PUBMED_CSV))
    assert docs and all(d["source"] == "pubmed" for d in docs)

    tfidf.upload_to_qdrant(tfidf.setup_collection(docs))
    assert tfidf.get_collection_info()["vectors_count"] == len(docs)

    results = tfidf.search_similar_documents("hypoxaemia in children", limit=3)
    assert len(results) == 3
    assert "hypoxaemia" in results[0]["text"].lower()


def test_milvus_lite_round_trip(tmp_path):
    from pymilvus import MilvusClient

    client = MilvusClient(uri=str(tmp_path / "milvus.db"))
    client.create_collection("docs", dimension=4)
    client.insert(
        "docs", [{"id": i, "vector": [float(i), 1.0, 0.0, 0.0]} for i in range(3)]
    )
    hits = client.search("docs", data=[[2.0, 1.0, 0.0, 0.0]], limit=1)
    assert hits[0][0]["id"] == 2


@pytest.mark.parametrize(
    ("uri", "expected"),
    [("artifacts/milvus.db", True), ("http://localhost:19530", False), (None, False)],
)
def test_is_lite_uri(uri, expected):
    assert SciNCLIngestion.is_lite_uri(uri) is expected
