import json
from pathlib import Path

import pytest

from rag_lab.cli import evaluate
from rag_lab.core import DATA_DIR, KnowledgeBase, generate_with_api


@pytest.fixture(scope="module")
def kb():
    return KnowledgeBase(threshold=0.08)


def test_hybrid_retrieval_returns_relevant_citable_document(kb):
    result = kb.ask("年假申请要经过谁审批？")
    assert result["supported"] is True
    assert result["evidence"][0]["source"] == "leave_policy.md"
    assert result["evidence"][0]["citation"] == "E1"
    assert "直属负责人审批" in result["answer"]


@pytest.mark.parametrize("question", ["这个实验室今年利润是多少？", "松桥实验室的 CEO 是谁？"])
def test_unanswerable_question_abstains_without_citation(kb, question):
    result = kb.ask(question)
    assert result["supported"] is False
    assert result["evidence"] == []
    assert "没有足够证据" in result["answer"]


def test_retrieval_evaluation_reports_separate_retrieval_only_metrics(kb, tmp_path):
    result = evaluate(kb, DATA_DIR / "eval.json", top_k=3)
    assert result["cases"] == 10
    assert result["answerable_cases"] == 8
    assert 0 <= result["hit_at_k"] <= 1
    assert 0 <= result["abstention_accuracy"] <= 1
    assert "excludes indexing" in result["scope"]
    assert all("retrieval_ms" in row for row in result["results"])


def test_api_generation_needs_explicit_configuration(monkeypatch):
    monkeypatch.delenv("RAG_API_KEY", raising=False)
    monkeypatch.delenv("RAG_MODEL", raising=False)
    with pytest.raises(ValueError, match="requires RAG_API_KEY"):
        generate_with_api("问题", [])
