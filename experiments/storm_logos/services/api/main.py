"""Storm-Logos API: FastAPI backend for therapy and dream analysis.

Run with:
    uvicorn storm_logos.services.api.main:app --reload --port 8000

Or in Docker:
    docker-compose up api
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure storm_logos is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from .deps import load_env, get_user_graph, get_dream_engine, get_semantic_data
from .routers import auth_router, sessions_router, evolution_router

# Load environment
load_env()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    print("Starting Storm-Logos API...")

    # Initialize services
    print("  Loading semantic data...")
    data = get_semantic_data()
    print(f"    {data.n_coordinates:,} coordinates")

    print("  Connecting to Neo4j...")
    ug = get_user_graph()
    if ug._connected:
        print("    Connected")
    else:
        print("    Warning: Neo4j not connected")

    print("  Initializing DreamEngine...")
    engine = get_dream_engine()
    print(f"    Model: {engine.model}")

    print("Storm-Logos API ready!")

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Storm-Logos API",
    description="""
Therapy and Dream Analysis API powered by semantic coordinates and Jungian archetypes.

## Features

- **Authentication**: Register and login to track your evolution
- **Sessions**: Start therapy or dream exploration sessions
- **Evolution**: Track how your archetypes manifest over time

## Archetypes

- shadow: The repressed, unknown aspects of self
- anima_animus: The contrasexual aspect of the psyche
- self: Wholeness and integration
- mother: Nurturing/devouring maternal principle
- father: Authority, order, spiritual principle
- hero: The ego's journey toward individuation
- trickster: Agent of change, boundary-crossing
- death_rebirth: Transformation through symbolic death
""",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(sessions_router)
app.include_router(evolution_router)


@app.get("/")
async def root():
    """API root - health check."""
    return {
        "service": "storm-logos",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/info")
async def info():
    """Get API info including model."""
    engine = get_dream_engine()
    return {
        "service": "storm-logos",
        "model": engine.model,
        "status": "running",
    }


@app.get("/corpus/books")
async def get_corpus_books():
    """Get list of processed books in corpus."""
    try:
        from storm_logos.data.neo4j import get_neo4j
        neo4j = get_neo4j()
        if not neo4j.connect():
            return {"books": [], "total": 0, "error": "Neo4j not connected"}

        # Try to get Book nodes directly first
        detailed_books = []
        with neo4j._driver.session() as session:
            # Query all Book nodes
            query = """
            MATCH (b:Book)
            OPTIONAL MATCH (b)-[:CONTAINS]->(bond:Bond)
            RETURN b.id as id, b.title as title, b.author as author,
                   b.genre as genre, count(bond) as n_bonds
            ORDER BY b.author, b.title
            """
            result = session.run(query)
            for record in result:
                book_id = record["id"] or "unknown"
                detailed_books.append({
                    "id": book_id,
                    "title": record["title"] or book_id.replace("_", " ").title() if book_id else "Unknown",
                    "author": record["author"] or "Unknown",
                    "n_bonds": record["n_bonds"] or 0,
                    "genre": record["genre"] or ""
                })

        # If no Book nodes, check FOLLOWS edges for book IDs
        if not detailed_books:
            books = neo4j.get_books()
            for book_id in books:
                if book_id:
                    detailed_books.append({
                        "id": book_id,
                        "title": book_id.replace("_", " ").title(),
                        "author": "Unknown",
                        "n_bonds": 0,
                        "genre": ""
                    })

        return {"books": detailed_books, "total": len(detailed_books)}
    except Exception as e:
        import traceback
        return {"books": [], "total": 0, "error": str(e), "trace": traceback.format_exc()}


@app.post("/corpus/process")
async def process_book_text(data: dict):
    """Process book text and load into corpus."""
    text = data.get("text", "")
    title = data.get("title", "Untitled")
    author = data.get("author", "Unknown")

    if not text or len(text) < 100:
        return {"error": "Text too short (min 100 chars)"}

    try:
        from storm_logos.data.book_parser import BookParser
        from storm_logos.data.neo4j import get_neo4j
        from storm_logos.data.postgres import get_data

        # Parse text
        parser = BookParser()
        parsed = parser.parse_text(text, title=title, author=author)

        if not parsed.bonds:
            return {"error": "No bonds extracted from text"}

        # Get coordinates
        data_layer = get_data()
        bonds_with_coords = []
        for eb in parsed.bonds:
            bond_id = f"{eb.adj}_{eb.noun}"
            adj_coords = data_layer.get(eb.adj)
            noun_coords = data_layer.get(eb.noun)

            if adj_coords and noun_coords:
                from storm_logos.data.models import Bond
                bond = Bond(
                    id=bond_id,
                    adj=eb.adj,
                    noun=eb.noun,
                    A=(adj_coords.A + noun_coords.A) / 2,
                    S=(adj_coords.S + noun_coords.S) / 2,
                    tau=(adj_coords.tau + noun_coords.tau) / 2
                )
                bonds_with_coords.append((eb, bond))

        # Store in Neo4j
        neo4j = get_neo4j()
        if not neo4j.connect():
            return {"error": "Neo4j not connected"}

        book_id = f"{author.lower().replace(' ', '_')}_{title.lower().replace(' ', '_')}"

        with neo4j._driver.session() as session:
            # Create author and book nodes
            session.run("""
                MERGE (a:Author {name: $author})
                MERGE (b:Book {id: $book_id})
                SET b.title = $title, b.author = $author,
                    b.n_bonds = $n_bonds, b.n_sentences = $n_sentences
                MERGE (a)-[:WROTE]->(b)
            """, author=author, book_id=book_id, title=title,
                n_bonds=len(bonds_with_coords), n_sentences=parsed.n_sentences)

            # Create bonds and FOLLOWS edges
            prev_bond_id = None
            for i, (eb, bond) in enumerate(bonds_with_coords):
                session.run("""
                    MERGE (bond:Bond {id: $bond_id})
                    SET bond.adj = $adj, bond.noun = $noun,
                        bond.A = $A, bond.S = $S, bond.tau = $tau
                    WITH bond
                    MATCH (b:Book {id: $book_id})
                    MERGE (b)-[:CONTAINS {chapter: $chapter, sentence: $sentence}]->(bond)
                """, bond_id=bond.id, adj=bond.adj, noun=bond.noun,
                    A=bond.A, S=bond.S, tau=bond.tau,
                    book_id=book_id, chapter=eb.chapter, sentence=eb.sentence)

                # Create FOLLOWS edge
                if prev_bond_id:
                    session.run("""
                        MATCH (b1:Bond {id: $prev_id}), (b2:Bond {id: $curr_id})
                        MERGE (b1)-[:FOLLOWS {book_id: $book_id, weight: 1.0}]->(b2)
                    """, prev_id=prev_bond_id, curr_id=bond.id, book_id=book_id)

                prev_bond_id = bond.id

        return {
            "success": True,
            "book_id": book_id,
            "title": title,
            "author": author,
            "n_bonds": len(bonds_with_coords),
            "n_sentences": parsed.n_sentences
        }

    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
