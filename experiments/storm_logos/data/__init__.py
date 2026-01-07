"""Data Layer: Storage and access to semantic data.

Provides:
    - Bond: noun-adjective pair with coordinates
    - WordCoordinates: (A, S, τ) position for a word
    - Trajectory: sequence of bonds
    - PostgresData: PostgreSQL connection for bonds/coordinates
    - Neo4jData: Neo4j connection for trajectories
    - BookParser: spaCy-based book parser
    - BookProcessor: Process books into Neo4j
"""

from .models import Bond, WordCoordinates, Trajectory, SemanticState
from .postgres import PostgresData, get_data
from .cache import CoordinateCache
from .neo4j import Neo4jData, Author, Book, get_neo4j
from .book_parser import BookParser, BookProcessor, ParsedBook, ExtractedBond
