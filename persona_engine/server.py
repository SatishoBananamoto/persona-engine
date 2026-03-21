"""
FastAPI Reference Server — REST API for the Persona Engine.

Provides HTTP endpoints for persona management, chat, planning, and validation.
Designed as a reference implementation for integrating the persona engine into
web services.

Usage:
    pip install persona-engine[server]
    uvicorn persona_engine.server:app --reload

    # Or run directly:
    python -m persona_engine.server

Configuration via environment variables:
    PERSONA_API_KEY     — Required API key for authenticated endpoints.
                          If unset, auth is disabled (development mode).
    PERSONA_MAX_SESSIONS — Maximum concurrent sessions (default: 100).
    PERSONA_SESSION_TTL  — Session TTL in seconds (default: 3600).
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from persona_engine import __version__
from persona_engine.engine import PersonaEngine, ChatResult
from persona_engine.schema.ir_schema import ConversationGoal, InteractionMode
from persona_engine.schema.persona_schema import Persona

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.environ.get("PERSONA_API_KEY")
_MAX_SESSIONS: int = int(os.environ.get("PERSONA_MAX_SESSIONS", "100"))
_SESSION_TTL: int = int(os.environ.get("PERSONA_SESSION_TTL", "3600"))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Persona Engine API",
    description=(
        "REST API for the Universal Conversational Persona Engine. "
        "Create psychologically-grounded synthetic personas and interact "
        "with them through structured, traceable conversations."
    ),
    version=__version__,
)

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def _require_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str | None:
    """Validate API key if PERSONA_API_KEY is configured.

    When the env var is unset, auth is disabled (development mode).
    When set, all mutating endpoints require a matching X-API-Key header.
    """
    if _API_KEY is None:
        return None  # Auth disabled
    if not api_key or api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ---------------------------------------------------------------------------
# Rate Limiting (simple in-process, per-IP)
# ---------------------------------------------------------------------------

_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX = 60  # requests per window per IP

_rate_limit_lock = threading.Lock()
_rate_limit_store: dict[str, list[float]] = {}


def _check_rate_limit(request: Request) -> None:
    """Simple sliding-window rate limiter. No external dependencies."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    cutoff = now - _RATE_LIMIT_WINDOW

    with _rate_limit_lock:
        timestamps = _rate_limit_store.get(client_ip, [])
        # Trim expired entries
        timestamps = [t for t in timestamps if t > cutoff]
        if len(timestamps) >= _RATE_LIMIT_MAX:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded ({_RATE_LIMIT_MAX} requests per {_RATE_LIMIT_WINDOW}s)",
            )
        timestamps.append(now)
        _rate_limit_store[client_ip] = timestamps


# ---------------------------------------------------------------------------
# Thread-safe Session Store
# ---------------------------------------------------------------------------

_sessions_lock = threading.Lock()
_sessions: dict[str, PersonaEngine] = {}
_session_timestamps: dict[str, float] = {}  # session_id → last access time


def _evict_expired_sessions() -> None:
    """Remove sessions older than TTL. Must be called under _sessions_lock."""
    now = time.monotonic()
    expired = [
        sid for sid, ts in _session_timestamps.items()
        if now - ts > _SESSION_TTL
    ]
    for sid in expired:
        _sessions.pop(sid, None)
        _session_timestamps.pop(sid, None)


def _get_session(session_id: str) -> PersonaEngine:
    with _sessions_lock:
        _evict_expired_sessions()
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        _session_timestamps[session_id] = time.monotonic()  # refresh TTL
        return _sessions[session_id]


def _store_session(engine: PersonaEngine) -> str:
    """Store a new session with cap enforcement. Returns session_id."""
    with _sessions_lock:
        _evict_expired_sessions()
        if len(_sessions) >= _MAX_SESSIONS:
            raise HTTPException(
                status_code=503,
                detail=f"Maximum sessions ({_MAX_SESSIONS}) reached. Delete unused sessions.",
            )
        session_id = str(uuid.uuid4())[:8]
        _sessions[session_id] = engine
        _session_timestamps[session_id] = time.monotonic()
        return session_id


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    """Request to create a new conversation session."""
    persona_id: str | None = Field(
        None,
        description="Persona filename relative to the personas/ directory (e.g. 'chef.yaml')",
    )
    persona_data: dict[str, Any] | None = Field(
        None,
        description="Raw persona data dict (alternative to persona_id)",
    )
    llm_provider: str = Field(
        "template",
        description="LLM backend: 'mock', 'template', 'anthropic', 'openai'",
    )
    seed: int = Field(42, description="Random seed for deterministic behavior")
    strict_mode: bool = Field(False, description="Force deterministic template output")


class CreateSessionResponse(BaseModel):
    """Response with session ID."""
    session_id: str
    persona_label: str
    persona_id: str


class ChatRequest(BaseModel):
    """Request to send a message."""
    message: str = Field(..., min_length=1, max_length=10_000)
    interaction_mode: str | None = None
    goal: str | None = None
    topic: str | None = None


class IRSummary(BaseModel):
    """Summarized IR for API responses."""
    confidence: float
    competence: float
    elasticity: float
    tone: str
    verbosity: str
    formality: float
    directness: float
    disclosure_level: float
    intent: str
    stance: str
    citation_count: int


class ValidationSummary(BaseModel):
    """Summarized validation result."""
    passed: bool
    violation_count: int
    violations: list[str]


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    text: str
    turn_number: int
    ir: IRSummary
    validation: ValidationSummary


class PlanResponse(BaseModel):
    """Response from plan endpoint (IR only, no text)."""
    turn_number: int
    ir: IRSummary


class SessionInfoResponse(BaseModel):
    """Session status information."""
    session_id: str
    persona_label: str
    persona_id: str
    turn_count: int


class PersonaListItem(BaseModel):
    """Single persona in the library listing."""
    filename: str
    persona_id: str
    label: str
    occupation: str
    age: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    active_sessions: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ir_to_summary(ir) -> IRSummary:
    return IRSummary(
        confidence=ir.response_structure.confidence,
        competence=ir.response_structure.competence,
        elasticity=ir.response_structure.elasticity,
        tone=ir.communication_style.tone.value,
        verbosity=ir.communication_style.verbosity.value,
        formality=ir.communication_style.formality,
        directness=ir.communication_style.directness,
        disclosure_level=ir.knowledge_disclosure.disclosure_level,
        intent=ir.response_structure.intent,
        stance=ir.response_structure.stance,
        citation_count=len(ir.citations),
    )


def _validation_to_summary(validation) -> ValidationSummary:
    violations = []
    if hasattr(validation, "violations") and validation.violations:
        violations = [v.message for v in validation.violations]
    return ValidationSummary(
        passed=validation.passed,
        violation_count=len(violations),
        violations=violations,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint (no auth required)."""
    return HealthResponse(
        status="ok",
        version=__version__,
        active_sessions=len(_sessions),
    )


@app.post("/sessions", response_model=CreateSessionResponse, status_code=201,
          dependencies=[Depends(_check_rate_limit)])
def create_session(
    req: CreateSessionRequest,
    _key: str | None = Depends(_require_api_key),
):
    """Create a new conversation session with a persona."""
    if req.persona_id and req.persona_data:
        raise HTTPException(
            status_code=400,
            detail="Provide either persona_id or persona_data, not both",
        )

    if not req.persona_id and not req.persona_data:
        raise HTTPException(
            status_code=400,
            detail="Must provide persona_id (YAML path) or persona_data (dict)",
        )

    try:
        if req.persona_id:
            # Validate persona path to prevent path traversal attacks.
            # Resolve against the personas directory and verify the result
            # stays within it.
            personas_dir = Path("personas").resolve()
            requested = (personas_dir / req.persona_id).resolve()
            if not str(requested).startswith(str(personas_dir) + "/"):
                raise HTTPException(
                    status_code=400,
                    detail="Invalid persona_id: path traversal not allowed",
                )
            engine = PersonaEngine.from_yaml(
                str(requested),
                llm_provider=req.llm_provider,
                seed=req.seed,
                strict_mode=req.strict_mode,
            )
        else:
            persona = Persona(**req.persona_data)  # type: ignore[arg-type]
            engine = PersonaEngine(
                persona,
                llm_provider=req.llm_provider,
                seed=req.seed,
                strict_mode=req.strict_mode,
            )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Persona file not found: {req.persona_id}")
    except HTTPException:
        raise  # Re-raise HTTPExceptions (path traversal, session cap)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    session_id = _store_session(engine)

    return CreateSessionResponse(
        session_id=session_id,
        persona_label=engine.persona.label,
        persona_id=engine.persona.persona_id,
    )


@app.post("/sessions/{session_id}/chat", response_model=ChatResponse,
          dependencies=[Depends(_check_rate_limit)])
def chat(
    session_id: str,
    req: ChatRequest,
    _key: str | None = Depends(_require_api_key),
):
    """Send a message and get a full response (IR + text + validation)."""
    engine = _get_session(session_id)

    kwargs: dict[str, Any] = {}
    if req.interaction_mode:
        try:
            kwargs["mode"] = InteractionMode(req.interaction_mode)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interaction_mode: {req.interaction_mode}",
            )
    if req.goal:
        try:
            kwargs["goal"] = ConversationGoal(req.goal)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid goal: {req.goal}")
    if req.topic:
        kwargs["topic"] = req.topic

    try:
        result = engine.chat(req.message, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ChatResponse(
        text=result.text,
        turn_number=result.turn_number,
        ir=_ir_to_summary(result.ir),
        validation=_validation_to_summary(result.validation),
    )


@app.post("/sessions/{session_id}/plan", response_model=PlanResponse,
          dependencies=[Depends(_check_rate_limit)])
def plan(
    session_id: str,
    req: ChatRequest,
    _key: str | None = Depends(_require_api_key),
):
    """Generate IR only (no LLM call). Useful for debugging and testing."""
    engine = _get_session(session_id)

    kwargs: dict[str, Any] = {}
    if req.interaction_mode:
        try:
            kwargs["mode"] = InteractionMode(req.interaction_mode)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid interaction_mode: {req.interaction_mode}",
            )
    if req.goal:
        try:
            kwargs["goal"] = ConversationGoal(req.goal)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid goal: {req.goal}")
    if req.topic:
        kwargs["topic"] = req.topic

    try:
        ir = engine.plan(req.message, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return PlanResponse(
        turn_number=engine.turn_count,
        ir=_ir_to_summary(ir),
    )


@app.get("/sessions/{session_id}", response_model=SessionInfoResponse)
def get_session_info(
    session_id: str,
    _key: str | None = Depends(_require_api_key),
):
    """Get information about an active session."""
    engine = _get_session(session_id)
    return SessionInfoResponse(
        session_id=session_id,
        persona_label=engine.persona.label,
        persona_id=engine.persona.persona_id,
        turn_count=engine.turn_count,
    )


@app.post("/sessions/{session_id}/reset")
def reset_session(
    session_id: str,
    _key: str | None = Depends(_require_api_key),
):
    """Reset session state (clears memory, resets turn counter)."""
    engine = _get_session(session_id)
    engine.reset()
    return {"status": "ok", "message": "Session reset"}


@app.delete("/sessions/{session_id}")
def delete_session(
    session_id: str,
    _key: str | None = Depends(_require_api_key),
):
    """Delete a session."""
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
        del _sessions[session_id]
        _session_timestamps.pop(session_id, None)
    return {"status": "ok", "message": "Session deleted"}


@app.get("/sessions", response_model=list[SessionInfoResponse])
def list_sessions(_key: str | None = Depends(_require_api_key)):
    """List all active sessions."""
    with _sessions_lock:
        _evict_expired_sessions()
        return [
            SessionInfoResponse(
                session_id=sid,
                persona_label=engine.persona.label,
                persona_id=engine.persona.persona_id,
                turn_count=engine.turn_count,
            )
            for sid, engine in _sessions.items()
        ]


@app.get("/personas", response_model=list[PersonaListItem])
def list_personas():
    """List available persona YAML files (no auth required)."""
    personas_dir = Path("personas")
    if not personas_dir.exists():
        return []

    results = []
    import yaml  # type: ignore[import-untyped]

    for f in sorted(personas_dir.glob("*.yaml")):
        try:
            with open(f) as fh:
                data = yaml.safe_load(fh)
            results.append(PersonaListItem(
                filename=f.name,
                persona_id=data.get("persona_id", "unknown"),
                label=data.get("label", f.stem),
                occupation=data.get("identity", {}).get("occupation", "unknown"),
                age=data.get("identity", {}).get("age", 0),
            ))
        except Exception:
            continue
    return results


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("persona_engine.server:app", host="0.0.0.0", port=8000, reload=True)
