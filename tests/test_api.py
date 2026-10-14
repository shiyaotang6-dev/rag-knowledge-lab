from fastapi.testclient import TestClient

from rag_lab.api import app


def test_browser_and_ask_api_work_without_keys():
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["documents"] == 8
        response = client.post("/api/ask", json={"question": "差旅报销在返程后多久提交？"})
        assert response.status_code == 200
        assert response.json()["evidence"][0]["source"] == "expense_policy.md"
        assert "十个工作日" in response.json()["answer"]


def test_invalid_and_unconfigured_requests_are_explicit():
    with TestClient(app) as client:
        assert client.post("/api/ask", json={"question": "   "}).status_code == 422
        assert client.post("/api/ask", json={"question": "怎么申请年假？", "mode": "deepseek"}).status_code == 409
