"""
Tests for the FastAPI reference server.

Uses FastAPI's TestClient for synchronous testing without starting a server.
"""

import pytest
from fastapi.testclient import TestClient

from persona_engine.server import app, _sessions


@pytest.fixture(autouse=True)
def clear_sessions():
    """Clear session store before each test."""
    _sessions.clear()
    yield
    _sessions.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def session_id(client):
    """Create a session and return its ID."""
    resp = client.post("/sessions", json={
        "persona_id": "chef.yaml",
        "llm_provider": "mock",
    })
    assert resp.status_code == 201
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.4.0"
        assert data["active_sessions"] == 0


# ---------------------------------------------------------------------------
# Session Management
# ---------------------------------------------------------------------------


class TestSessions:
    def test_create_session_from_yaml(self, client):
        resp = client.post("/sessions", json={
            "persona_id": "chef.yaml",
            "llm_provider": "mock",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["persona_label"]
        assert data["persona_id"]

    def test_create_session_from_data(self, client):
        import yaml
        with open("personas/chef.yaml") as f:
            persona_data = yaml.safe_load(f)
        resp = client.post("/sessions", json={
            "persona_data": persona_data,
            "llm_provider": "template",
        })
        assert resp.status_code == 201

    def test_create_session_no_persona_fails(self, client):
        resp = client.post("/sessions", json={
            "llm_provider": "mock",
        })
        assert resp.status_code == 400

    def test_create_session_both_persona_types_fails(self, client):
        resp = client.post("/sessions", json={
            "persona_id": "chef.yaml",
            "persona_data": {"some": "data"},
            "llm_provider": "mock",
        })
        assert resp.status_code == 400

    def test_create_session_nonexistent_file(self, client):
        resp = client.post("/sessions", json={
            "persona_id": "nonexistent.yaml",
            "llm_provider": "mock",
        })
        assert resp.status_code == 404

    def test_get_session_info(self, client, session_id):
        resp = client.get(f"/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["turn_count"] == 0

    def test_get_nonexistent_session(self, client):
        resp = client.get("/sessions/nonexistent")
        assert resp.status_code == 404

    def test_list_sessions(self, client, session_id):
        resp = client.get("/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["session_id"] == session_id

    def test_delete_session(self, client, session_id):
        resp = client.delete(f"/sessions/{session_id}")
        assert resp.status_code == 200
        # Verify it's gone
        resp = client.get(f"/sessions/{session_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_session(self, client):
        resp = client.delete("/sessions/nonexistent")
        assert resp.status_code == 404

    def test_reset_session(self, client, session_id):
        # Chat first
        client.post(f"/sessions/{session_id}/chat", json={"message": "Hello"})
        # Reset
        resp = client.post(f"/sessions/{session_id}/reset")
        assert resp.status_code == 200
        # Verify turn count reset
        info = client.get(f"/sessions/{session_id}").json()
        assert info["turn_count"] == 0


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestChat:
    def test_chat_basic(self, client, session_id):
        resp = client.post(f"/sessions/{session_id}/chat", json={
            "message": "What makes a perfect sauce?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"]
        assert data["turn_number"] == 1
        assert "confidence" in data["ir"]
        assert "tone" in data["ir"]
        assert "passed" in data["validation"]

    def test_chat_multi_turn(self, client, session_id):
        r1 = client.post(f"/sessions/{session_id}/chat", json={
            "message": "Tell me about sauces.",
        }).json()
        r2 = client.post(f"/sessions/{session_id}/chat", json={
            "message": "And soups?",
        }).json()
        assert r1["turn_number"] == 1
        assert r2["turn_number"] == 2

    def test_chat_empty_message(self, client, session_id):
        resp = client.post(f"/sessions/{session_id}/chat", json={
            "message": "",
        })
        assert resp.status_code == 422  # Pydantic validation

    def test_chat_nonexistent_session(self, client):
        resp = client.post("/sessions/nonexistent/chat", json={
            "message": "Hello",
        })
        assert resp.status_code == 404

    def test_chat_with_interaction_mode(self, client, session_id):
        resp = client.post(f"/sessions/{session_id}/chat", json={
            "message": "Tell me about your background",
            "interaction_mode": "interview",
        })
        assert resp.status_code == 200

    def test_ir_summary_fields(self, client, session_id):
        resp = client.post(f"/sessions/{session_id}/chat", json={
            "message": "What is your specialty?",
        })
        ir = resp.json()["ir"]
        assert 0.0 <= ir["confidence"] <= 1.0
        assert 0.0 <= ir["competence"] <= 1.0
        assert 0.0 <= ir["elasticity"] <= 1.0
        assert 0.0 <= ir["formality"] <= 1.0
        assert 0.0 <= ir["directness"] <= 1.0
        assert 0.0 <= ir["disclosure_level"] <= 1.0
        assert ir["citation_count"] > 0


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class TestPlan:
    def test_plan_basic(self, client, session_id):
        resp = client.post(f"/sessions/{session_id}/plan", json={
            "message": "What do you cook?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "ir" in data
        assert data["ir"]["confidence"] >= 0

    def test_plan_nonexistent_session(self, client):
        resp = client.post("/sessions/nonexistent/plan", json={
            "message": "Hello",
        })
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Persona Library
# ---------------------------------------------------------------------------


class TestPersonaLibrary:
    def test_list_personas(self, client):
        resp = client.get("/personas")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) > 0
        # Should include the chef
        filenames = [p["filename"] for p in data]
        assert "chef.yaml" in filenames

    def test_persona_list_item_fields(self, client):
        resp = client.get("/personas")
        data = resp.json()
        for p in data:
            assert "filename" in p
            assert "persona_id" in p
            assert "label" in p
            assert "occupation" in p
            assert "age" in p


# ---------------------------------------------------------------------------
# Strict Mode
# ---------------------------------------------------------------------------


class TestPathTraversalSecurity:
    """Verify that persona_id cannot be used to read arbitrary files."""

    def test_traversal_with_dotdot(self, client):
        resp = client.post("/sessions", json={
            "persona_id": "../../../etc/passwd",
            "llm_provider": "mock",
        })
        assert resp.status_code == 400
        assert "path traversal" in resp.json()["detail"].lower()

    def test_traversal_with_absolute_path(self, client):
        resp = client.post("/sessions", json={
            "persona_id": "/etc/passwd",
            "llm_provider": "mock",
        })
        assert resp.status_code == 400
        assert "path traversal" in resp.json()["detail"].lower()

    def test_traversal_with_nested_escape(self, client):
        resp = client.post("/sessions", json={
            "persona_id": "subdir/../../etc/passwd",
            "llm_provider": "mock",
        })
        assert resp.status_code == 400
        assert "path traversal" in resp.json()["detail"].lower()

    def test_valid_persona_id_still_works(self, client):
        resp = client.post("/sessions", json={
            "persona_id": "chef.yaml",
            "llm_provider": "mock",
        })
        assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Strict Mode
# ---------------------------------------------------------------------------


class TestStrictModeServer:
    def test_strict_mode_session(self, client):
        resp = client.post("/sessions", json={
            "persona_id": "chef.yaml",
            "llm_provider": "template",
            "strict_mode": True,
        })
        assert resp.status_code == 201
        sid = resp.json()["session_id"]

        resp = client.post(f"/sessions/{sid}/chat", json={
            "message": "What makes a good dish?",
        })
        assert resp.status_code == 200
        assert resp.json()["text"]
