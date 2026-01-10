"""Sessions Router: Therapy and dream session management."""

import json
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from fastapi import APIRouter, HTTPException, status, Depends

from ..models import (
    SessionStart, SessionMessage, SessionResponse, SessionEnd,
    SessionMode, ArchetypeManifestation
)
from ..deps import (
    get_current_user, get_optional_user, get_dream_engine, get_user_graph,
    get_session, store_session, remove_session, get_user_active_session
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("/history")
async def get_session_history(
    current_user: dict = Depends(get_current_user)
):
    """Get user's session history."""
    ug = get_user_graph()
    sessions = ug.get_user_sessions(current_user["user_id"])
    return {"sessions": sessions}


@dataclass
class SessionState:
    """Active session state."""
    session_id: str
    user_id: Optional[str]
    mode: str = "hybrid"
    turn: int = 0
    dream_text: Optional[str] = None
    symbols: List[Dict] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    emotions: List[str] = field(default_factory=list)
    history: List[Dict] = field(default_factory=list)
    started_at: str = ""


def _analyze_input(engine, text: str, state: SessionState) -> Dict[str, Any]:
    """Analyze user input to understand intent."""
    system = """Analyze the user's input in a therapy/dream context. Return JSON:
{
    "type": "greeting|dream_content|association|emotion|question|reflection|goodbye",
    "mode_hint": "dream|therapy|unclear",
    "contains_dream": true/false,
    "emotions_detected": ["list"],
    "key_symbols": ["list"],
    "summary": "brief summary"
}"""

    prompt = f"""Session mode: {state.mode}
Turn: {state.turn}

User input: "{text}"

Return only valid JSON."""

    try:
        response = engine._call_llm(system, prompt, max_tokens=200)
        if "{" in response:
            json_str = response[response.index("{"):response.rindex("}")+1]
            return json.loads(json_str)
    except:
        pass

    return {"type": "unclear", "mode_hint": "unclear", "contains_dream": False,
            "emotions_detected": [], "key_symbols": [], "summary": text[:100]}


def _generate_response(engine, text: str, state: SessionState, analysis: Dict) -> str:
    """Generate therapeutic response."""
    # Build full conversation history (not truncated!)
    history_context = ""
    if state.history:
        recent = state.history[-5:]  # Last 5 turns
        history_context = "\n".join([
            f"Turn {h.get('turn', '?')}:\nUser: {h.get('user', '')}\nTherapist: {h.get('therapist', '')}"
            for h in recent
        ])

    # Include dream text if available - this is the core content
    dream_context = ""
    if state.dream_text:
        dream_context = f"\n\nTHE DREAM (shared earlier):\n{state.dream_text}\n"

    # Symbols collected so far
    symbols_context = ""
    if state.symbols:
        symbols_context = f"\nSymbols identified: {', '.join([s.get('text', '') for s in state.symbols[:10]])}"

    system = f"""You are a depth psychologist conducting a {state.mode} session.

Your approach:
1. Listen deeply and reflect what you hear
2. Ask focused questions - one at a time
3. Connect symbols to psychological meaning
4. Use Jungian concepts naturally: shadow, anima/animus, Self, archetypes
5. Keep responses concise (2-4 sentences)
6. NEVER ask the user to repeat what they already shared
{dream_context}{symbols_context}
{f"Conversation so far:{chr(10)}{history_context}" if history_context else ""}"""

    prompt = f"""User says: "{text}"

Analysis: {analysis.get('summary', '')}
Emotions: {', '.join(analysis.get('emotions_detected', []))}

Respond therapeutically. Remember: you already have the dream content above - do not ask for it again."""

    return engine._call_llm(system, prompt, max_tokens=300)


def _extract_archetypes(engine, state: SessionState) -> List[Dict]:
    """Extract archetypes qualitatively from session."""
    if not state.history:
        return []

    all_text = " ".join([
        h.get("user", "") + " " + h.get("therapist", "")
        for h in state.history
    ])

    system = """Analyze this session and identify Jungian archetypes that manifested.
Return JSON array:
[{"archetype": "shadow|anima_animus|self|mother|father|hero|trickster|death_rebirth",
  "symbols": ["symbol1"], "emotions": ["emotion1"], "context": "brief description"}]
Only include clearly present archetypes. Return [] if none."""

    symbol_text = f"\nSymbols: {', '.join([s.get('text', '') for s in state.symbols[:10]])}" if state.symbols else ""

    prompt = f"""Session content:
{all_text[:2500]}
{symbol_text}

Extract archetypes. Return only valid JSON array."""

    try:
        response = engine._call_llm(system, prompt, max_tokens=400)
        if "[" in response:
            json_str = response[response.index("["):response.rindex("]")+1]
            return json.loads(json_str)
    except:
        pass

    return []


@router.post("/start", response_model=SessionResponse)
async def start_session(
    data: SessionStart = None,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """Start a new therapy/dream session."""
    user_id = current_user["user_id"] if current_user else None

    # Check for existing session
    if user_id:
        existing = get_user_active_session(user_id)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"You have an active session: {existing}. End it first."
            )

    # Create session
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if user_id:
        session_id = f"{user_id[:8]}_{session_id}"

    mode = data.mode.value if data and data.mode else "hybrid"

    state = SessionState(
        session_id=session_id,
        user_id=user_id,
        mode=mode,
        started_at=datetime.now().isoformat(),
    )
    store_session(session_id, state)

    # Welcome message
    if mode == "dream":
        welcome = "Welcome to dream exploration. Share a dream you'd like to understand."
    elif mode == "therapy":
        welcome = "I'm here to listen. What's on your mind today?"
    else:
        welcome = "Welcome. You can share a dream, talk about what's weighing on you, or start wherever feels right."

    return SessionResponse(
        session_id=session_id,
        mode=SessionMode(mode),
        turn=0,
        response=welcome,
        symbols=[],
        emotions=[],
        themes=[],
    )


@router.post("/{session_id}/message", response_model=SessionResponse)
async def send_message(
    session_id: str,
    data: SessionMessage,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """Send a message in the session."""
    state = get_session(session_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership if authenticated
    if current_user and state.user_id and state.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This is not your session"
        )

    engine = get_dream_engine()
    user_input = data.message.strip()

    # Analyze input
    analysis = _analyze_input(engine, user_input, state)

    # Check for goodbye
    if analysis.get("type") == "goodbye":
        return await end_session(session_id, current_user)

    # Update mode if needed
    if state.mode == "hybrid":
        if analysis.get("contains_dream") or analysis.get("mode_hint") == "dream":
            state.mode = "dream"
        elif analysis.get("mode_hint") == "therapy":
            state.mode = "therapy"

    # Extract symbols if dream content
    if analysis.get("contains_dream") or analysis.get("type") == "dream_content":
        if not state.dream_text:
            state.dream_text = user_input
        new_symbols = engine.extract_symbols(user_input)
        for s in new_symbols:
            state.symbols.append({
                "text": s.raw_text,
                "archetype": s.archetype,
                "A": s.bond.A,
                "S": s.bond.S,
            })

    # Track emotions and themes
    if analysis.get("emotions_detected"):
        state.emotions.extend(analysis["emotions_detected"])
    if analysis.get("key_symbols"):
        state.themes.extend(analysis["key_symbols"])

    # Generate response
    response = _generate_response(engine, user_input, state, analysis)

    # Update state
    state.turn += 1
    state.history.append({
        "turn": state.turn,
        "user": user_input,
        "therapist": response,
        "timestamp": datetime.now().isoformat(),
    })

    store_session(session_id, state)

    return SessionResponse(
        session_id=session_id,
        mode=SessionMode(state.mode),
        turn=state.turn,
        response=response,
        symbols=state.symbols[-5:],  # Last 5
        emotions=list(set(state.emotions[-5:])),
        themes=list(set(state.themes[-5:])),
    )


@router.post("/{session_id}/end", response_model=SessionEnd)
async def end_session(
    session_id: str,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """End a session and extract archetypes."""
    state = get_session(session_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    engine = get_dream_engine()

    # Extract archetypes
    archetypes = _extract_archetypes(engine, state)

    # Save to user graph if authenticated
    if state.user_id:
        try:
            from storm_logos.data.user_graph import SessionRecord, ArchetypeManifestation as AM

            manifestations = [
                AM(
                    archetype=a.get("archetype", "unknown"),
                    symbols=a.get("symbols", []),
                    emotions=a.get("emotions", []),
                    context=a.get("context", ""),
                )
                for a in archetypes
            ]

            record = SessionRecord(
                session_id=session_id,
                user_id=state.user_id,
                mode=state.mode,
                timestamp=state.started_at,
                dream_text=state.dream_text,
                archetypes=manifestations,
                symbols=[s.get("text", "") for s in state.symbols],
                emotions=list(set(state.emotions)),
                themes=list(set(state.themes)),
                summary=f"{state.turn} turns",
            )

            ug = get_user_graph()
            ug.save_session(record)
        except Exception as e:
            print(f"Warning: Could not save to user graph: {e}")

    # Remove from active sessions
    remove_session(session_id)

    return SessionEnd(
        session_id=session_id,
        turns=state.turn,
        mode=state.mode,
        symbols=state.symbols,
        emotions=list(set(state.emotions)),
        themes=list(set(state.themes)),
        archetypes=archetypes,
        summary=f"Session completed. {state.turn} turns, {len(state.symbols)} symbols, {len(archetypes)} archetypes.",
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session_state(
    session_id: str,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """Get current session state."""
    state = get_session(session_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    return SessionResponse(
        session_id=session_id,
        mode=SessionMode(state.mode),
        turn=state.turn,
        response="",
        symbols=state.symbols[-10:],
        emotions=list(set(state.emotions[-10:])),
        themes=list(set(state.themes[-10:])),
    )


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    current_user: Optional[dict] = Depends(get_optional_user)
):
    """Delete a session without saving (discard)."""
    state = get_session(session_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Verify ownership if authenticated
    if current_user and state.user_id and state.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This is not your session"
        )

    # Just remove from active sessions - don't save anything
    remove_session(session_id)

    return {"message": "Session deleted", "session_id": session_id}


@router.post("/{session_id}/pause")
async def pause_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Pause a session (save to Neo4j without ending)."""
    state = get_session(session_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    if state.user_id != current_user["user_id"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="This is not your session"
        )

    # Save to Neo4j with paused status
    try:
        from storm_logos.data.user_graph import SessionRecord

        # Convert symbols to the right format
        symbols_list = [s.get("text", "") if isinstance(s, dict) else s for s in state.symbols]

        record = SessionRecord(
            session_id=session_id,
            user_id=state.user_id,
            mode=state.mode,
            timestamp=state.started_at,
            dream_text=state.dream_text,
            archetypes=[],
            symbols=symbols_list,
            emotions=list(set(state.emotions)),
            themes=list(set(state.themes)),
            history=state.history,
            summary=f"{state.turn} turns (paused)",
            status="paused",
        )

        ug = get_user_graph()
        ug.save_session(record)
    except Exception as e:
        print(f"Warning: Could not save session: {e}")

    # Remove from active sessions
    remove_session(session_id)

    return {"message": "Session paused", "session_id": session_id, "turns": state.turn}


@router.post("/{session_id}/resume", response_model=SessionResponse)
async def resume_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Resume a paused session."""
    # Check if already active
    existing = get_session(session_id)
    if existing:
        return SessionResponse(
            session_id=session_id,
            mode=SessionMode(existing.mode),
            turn=existing.turn,
            response="Session already active.",
            symbols=existing.symbols[-5:],
            emotions=list(set(existing.emotions[-5:])),
            themes=list(set(existing.themes[-5:])),
        )

    # Load from Neo4j
    ug = get_user_graph()
    data = ug.load_session(session_id, current_user["user_id"])

    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )

    # Convert symbols back to dict format
    symbols = []
    for s in data.get("symbols", []):
        if isinstance(s, str):
            symbols.append({"text": s, "archetype": None, "A": 0, "S": 0})
        else:
            symbols.append(s)

    # Reconstruct session state
    state = SessionState(
        session_id=session_id,
        user_id=current_user["user_id"],
        mode=data.get("mode", "hybrid"),
        turn=len(data.get("history", [])),
        dream_text=data.get("dream_text"),
        symbols=symbols,
        themes=data.get("themes", []),
        emotions=data.get("emotions", []),
        history=data.get("history", []),
        started_at=data.get("timestamp", datetime.now().isoformat()),
    )

    store_session(session_id, state)

    # Update status in Neo4j
    ug.update_session_status(session_id, "active")

    return SessionResponse(
        session_id=session_id,
        mode=SessionMode(state.mode),
        turn=state.turn,
        response=f"Session resumed. We were at turn {state.turn}. Continue where we left off.",
        symbols=state.symbols[-5:],
        emotions=list(set(state.emotions[-5:])),
        themes=list(set(state.themes[-5:])),
    )
