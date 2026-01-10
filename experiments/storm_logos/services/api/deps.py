"""API Dependencies: Shared state and authentication."""

import os
import jwt
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path

# JWT settings
JWT_SECRET = os.environ.get("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

security = HTTPBearer()


def load_env():
    """Load environment variables from .env files."""
    locations = [
        Path(__file__).parent.parent.parent / '.env',
        Path.home() / '.env',
        Path.cwd() / '.env',
    ]
    for env_path in locations:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if not os.environ.get(key):
                            os.environ[key] = value


# Load on import
load_env()


# =============================================================================
# JWT TOKENS
# =============================================================================

def create_token(user_id: str, username: str) -> str:
    """Create JWT token for user."""
    payload = {
        "user_id": user_id,
        "username": username,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> Dict[str, Any]:
    """Decode and validate JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )


# =============================================================================
# DEPENDENCIES
# =============================================================================

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, str]:
    """Get current user from JWT token."""
    token = credentials.credentials
    payload = decode_token(token)
    return {
        "user_id": payload["user_id"],
        "username": payload["username"],
    }


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[Dict[str, str]]:
    """Get current user if authenticated, None otherwise."""
    if not credentials:
        return None
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None


# =============================================================================
# SHARED STATE (singleton services)
# =============================================================================

_user_graph = None
_dream_engine = None
_data = None


def get_user_graph():
    """Get UserGraph singleton."""
    global _user_graph
    if _user_graph is None:
        from storm_logos.data.user_graph import get_user_graph as _get_ug
        _user_graph = _get_ug()
        _user_graph.connect()
    return _user_graph


def get_dream_engine():
    """Get DreamEngine singleton."""
    global _dream_engine
    if _dream_engine is None:
        model = os.environ.get("LLM_MODEL", "groq:llama-3.3-70b-versatile")
        from storm_logos.applications import DreamEngine
        _dream_engine = DreamEngine(model=model)
        _dream_engine.connect()
    return _dream_engine


def get_semantic_data():
    """Get semantic data singleton."""
    global _data
    if _data is None:
        from storm_logos.data.postgres import get_data
        _data = get_data()
    return _data


# =============================================================================
# SESSION STORAGE (in-memory for now, could use Redis)
# =============================================================================

# Active sessions: {session_id: SessionState}
_active_sessions: Dict[str, Any] = {}


def get_session(session_id: str) -> Optional[Any]:
    """Get active session by ID."""
    return _active_sessions.get(session_id)


def store_session(session_id: str, session_state: Any):
    """Store active session."""
    _active_sessions[session_id] = session_state


def remove_session(session_id: str):
    """Remove session from active sessions."""
    _active_sessions.pop(session_id, None)


def get_user_active_session(user_id: str) -> Optional[str]:
    """Get user's active session ID if any."""
    for sid, state in _active_sessions.items():
        if hasattr(state, 'user_id') and state.user_id == user_id:
            return sid
    return None
