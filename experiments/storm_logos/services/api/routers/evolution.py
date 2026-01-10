"""Evolution Router: Archetype evolution and user profile."""

from typing import List
from fastapi import APIRouter, HTTPException, status, Depends

from ..models import (
    UserProfile, ArchetypeEvolution, SessionHistory,
    DreamAnalysisRequest, DreamAnalysisResponse, DreamSymbol, ArchetypeManifestation
)
from ..deps import get_current_user, get_user_graph, get_dream_engine

router = APIRouter(prefix="/evolution", tags=["evolution"])


@router.get("/profile", response_model=UserProfile)
async def get_profile(current_user: dict = Depends(get_current_user)):
    """Get user's archetype profile."""
    ug = get_user_graph()
    profile = ug.get_user_archetype_profile(current_user["user_id"])

    return UserProfile(
        username=current_user["username"],
        total_sessions=profile.get("total_sessions", 0),
        archetypes=profile.get("archetypes", {}),
        dominant_archetypes=profile.get("dominant_archetypes", []),
    )


@router.get("/history", response_model=List[SessionHistory])
async def get_history(
    limit: int = 20,
    current_user: dict = Depends(get_current_user)
):
    """Get user's session history."""
    ug = get_user_graph()
    sessions = ug.get_user_sessions(current_user["user_id"], limit=limit)

    return [
        SessionHistory(
            session_id=s["session_id"],
            mode=s["mode"],
            timestamp=s["timestamp"],
            summary=s.get("summary", ""),
            archetypes=s.get("archetypes", []),
        )
        for s in sessions
    ]


@router.get("/archetype/{archetype}", response_model=List[ArchetypeEvolution])
async def get_archetype_evolution(
    archetype: str,
    current_user: dict = Depends(get_current_user)
):
    """Get evolution of a specific archetype over time."""
    valid_archetypes = [
        "shadow", "anima_animus", "self", "mother", "father",
        "hero", "trickster", "death_rebirth"
    ]

    if archetype not in valid_archetypes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid archetype. Must be one of: {', '.join(valid_archetypes)}"
        )

    ug = get_user_graph()
    evolution = ug.get_archetype_evolution(current_user["user_id"], archetype)

    return [
        ArchetypeEvolution(
            session_id=e["session_id"],
            timestamp=e["timestamp"],
            context=e.get("context", ""),
            symbols=e.get("symbols", []),
            emotions=e.get("emotions", []),
        )
        for e in evolution
    ]


@router.get("/symbols/recurring")
async def get_recurring_symbols(
    min_count: int = 2,
    current_user: dict = Depends(get_current_user)
):
    """Get symbols that recur across sessions."""
    ug = get_user_graph()
    symbols = ug.get_recurring_symbols(current_user["user_id"], min_count=min_count)

    return symbols


@router.get("/archetype/{archetype}/symbols")
async def get_archetype_symbols(
    archetype: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all symbols through which an archetype has manifested."""
    ug = get_user_graph()
    symbols = ug.get_archetype_symbols(current_user["user_id"], archetype)

    return symbols


@router.get("/emotional-patterns")
async def get_emotional_patterns(current_user: dict = Depends(get_current_user)):
    """Get which emotions associate with which archetypes."""
    ug = get_user_graph()
    patterns = ug.get_emotional_patterns(current_user["user_id"])

    return patterns


# =============================================================================
# DREAM ANALYSIS (standalone, doesn't require session)
# =============================================================================

@router.post("/analyze-dream", response_model=DreamAnalysisResponse)
async def analyze_dream(data: DreamAnalysisRequest):
    """Analyze a dream without starting a session."""
    engine = get_dream_engine()

    # Get full analysis
    analysis = engine.analyze(data.dream_text)

    # Convert symbols
    symbols = [
        DreamSymbol(
            text=s.raw_text,
            archetype=s.archetype,
            A=s.bond.A,
            S=s.bond.S,
            tau=s.bond.tau,
        )
        for s in analysis.symbols
    ]

    # Extract archetypes qualitatively
    archetypes = []
    state = analysis.state
    archetype_scores = {
        "shadow": state.shadow,
        "anima_animus": state.anima_animus,
        "self": state.self_archetype,
        "mother": state.mother,
        "father": state.father,
        "hero": state.hero,
        "trickster": state.trickster,
        "death_rebirth": state.death_rebirth,
    }

    # Get symbols for each archetype
    for arch_name, score in archetype_scores.items():
        if score > 0.3:  # Threshold
            arch_symbols = [s.raw_text for s in analysis.symbols if s.archetype == arch_name]
            archetypes.append(ArchetypeManifestation(
                archetype=arch_name,
                symbols=arch_symbols[:5],
                emotions=[],  # Could be extracted from text
                context=f"Score: {score:.2f}",
            ))

    dominant, _ = state.dominant_archetype()

    return DreamAnalysisResponse(
        symbols=symbols,
        archetypes=archetypes,
        dominant_archetype=dominant,
        coordinates={
            "A": state.A,
            "S": state.S,
            "tau": state.tau,
        },
        interpretation=analysis.interpretation,
        corpus_resonances=analysis.corpus_resonances,
    )
