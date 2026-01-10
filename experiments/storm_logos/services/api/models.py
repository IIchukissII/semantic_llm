"""API Models: Pydantic models for request/response."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class SessionMode(str, Enum):
    DREAM = "dream"
    THERAPY = "therapy"
    HYBRID = "hybrid"


# =============================================================================
# AUTH
# =============================================================================

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    username: str
    password: str


class UserResponse(BaseModel):
    user_id: str
    username: str
    created_at: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# =============================================================================
# SESSIONS
# =============================================================================

class SessionStart(BaseModel):
    mode: Optional[SessionMode] = None


class SessionMessage(BaseModel):
    message: str


class SessionResponse(BaseModel):
    session_id: str
    mode: SessionMode
    turn: int
    response: str
    symbols: List[Dict[str, Any]] = []
    emotions: List[str] = []
    themes: List[str] = []


class SessionEnd(BaseModel):
    session_id: str
    turns: int
    mode: str
    symbols: List[Dict[str, Any]]
    emotions: List[str]
    themes: List[str]
    archetypes: List[Dict[str, Any]]
    summary: str


# =============================================================================
# ARCHETYPES
# =============================================================================

class ArchetypeManifestation(BaseModel):
    archetype: str
    symbols: List[str]
    emotions: List[str]
    context: str = ""


class ArchetypeEvolution(BaseModel):
    session_id: str
    timestamp: str
    context: str
    symbols: List[str]
    emotions: List[str]


class UserProfile(BaseModel):
    username: str
    total_sessions: int
    archetypes: Dict[str, int]
    dominant_archetypes: List[str]


class SessionHistory(BaseModel):
    session_id: str
    mode: str
    timestamp: str
    summary: str
    archetypes: List[str]


# =============================================================================
# DREAM ANALYSIS
# =============================================================================

class DreamAnalysisRequest(BaseModel):
    dream_text: str


class DreamSymbol(BaseModel):
    text: str
    archetype: Optional[str] = None
    A: float = 0.0
    S: float = 0.0
    tau: float = 2.5


class DreamAnalysisResponse(BaseModel):
    symbols: List[DreamSymbol]
    archetypes: List[ArchetypeManifestation]
    dominant_archetype: str
    coordinates: Dict[str, float]
    interpretation: str
    corpus_resonances: List[Dict[str, str]] = []
