import pytest


@pytest.fixture(autouse=True)
def offline_test_environment(monkeypatch):
    """Never load remote models or invoke configured credentials during tests."""
    for name in ("RAG_API_KEY", "RAG_API_BASE", "RAG_MODEL", "RAG_EMBEDDING_MODEL", "RAG_RERANKER_MODEL"):
        monkeypatch.delenv(name, raising=False)
