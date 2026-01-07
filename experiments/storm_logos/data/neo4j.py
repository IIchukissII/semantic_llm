"""Neo4j Data Layer.

Connection to Neo4j for book trajectories and FOLLOWS edges.

Extended Schema:
    (:Author {name, era, domain})
    (:Book {id, title, author, filename, genre, processed_at, n_bonds, n_sentences})
    (:Bond {id, adj, noun, A, S, tau})

    (:Author)-[:WROTE]->(:Book)
    (:Book)-[:CONTAINS {chapter, sentence, position}]->(:Bond)
    (:Bond)-[:FOLLOWS {book_id, chapter, sentence, position}]->(:Bond)
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

from .models import Bond, Trajectory


@dataclass
class Author:
    """An author in the graph."""
    name: str
    era: str = ''  # e.g., 'ancient', 'modern', '20th_century'
    domain: str = ''  # e.g., 'psychology', 'mythology', 'fiction'


@dataclass
class Book:
    """A book in the graph."""
    id: str
    title: str
    author: str
    filename: str
    genre: str = ''  # e.g., 'psychology', 'mythology', 'epic'
    processed_at: Optional[datetime] = None
    n_bonds: int = 0
    n_sentences: int = 0
    n_chapters: int = 0
    metadata: Dict = field(default_factory=dict)


@dataclass
class FollowsEdge:
    """A FOLLOWS edge between bonds."""
    source_id: str
    target_id: str
    book_id: str
    chapter: int
    sentence: int
    position: int


class Neo4jData:
    """Neo4j connection for trajectories.

    Provides access to author-walked paths through semantic space.

    NOTE: This is a placeholder implementation.
    Full implementation requires neo4j-python-driver.
    """

    def __init__(self, uri: str = 'bolt://localhost:7687',
                 user: str = 'neo4j', password: str = 'experience123'):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Neo4j."""
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self.uri,
                auth=(self.user, self.password)
            )
            # Verify connection works
            self._driver.verify_connectivity()
            self._connected = True
            return True
        except ImportError:
            print("Neo4j driver not installed. Run: pip install neo4j")
            return False
        except Exception as e:
            print(f"Neo4j connection failed: {e}")
            return False

    def close(self):
        """Close connection."""
        if self._driver:
            self._driver.close()
            self._connected = False

    # ========================================================================
    # QUERIES
    # ========================================================================

    def get_followers(self, bond_id: str, limit: int = 50) -> List[Bond]:
        """Get bonds that FOLLOW this bond (where authors went next)."""
        if not self._connected:
            return []

        query = """
        MATCH (b:Bond {id: $bond_id})-[:FOLLOWS]->(next:Bond)
        RETURN next.adj, next.noun, next.A, next.S, next.tau
        LIMIT $limit
        """

        results = []
        with self._driver.session() as session:
            records = session.run(query, bond_id=bond_id, limit=limit)
            for record in records:
                results.append(Bond(
                    noun=record['next.noun'],
                    adj=record['next.adj'],
                    A=record['next.A'] or 0.0,
                    S=record['next.S'] or 0.0,
                    tau=record['next.tau'] or 2.5,
                ))

        return results

    def get_trajectory(self, book: str, start: int = 0,
                       length: int = 100) -> Trajectory:
        """Get a trajectory from a book."""
        if not self._connected:
            return Trajectory()

        query = """
        MATCH (b:Bond)-[f:FOLLOWS {book: $book}]->(next:Bond)
        WHERE f.position >= $start
        RETURN b.adj, b.noun, b.A, b.S, b.tau, f.position
        ORDER BY f.position
        LIMIT $length
        """

        trajectory = Trajectory(metadata={'book': book})
        with self._driver.session() as session:
            records = session.run(query, book=book, start=start, length=length)
            for record in records:
                trajectory.bonds.append(Bond(
                    noun=record['b.noun'],
                    adj=record['b.adj'],
                    A=record['b.A'] or 0.0,
                    S=record['b.S'] or 0.0,
                    tau=record['b.tau'] or 2.5,
                ))

        return trajectory

    def get_books(self) -> List[str]:
        """Get list of available books."""
        if not self._connected:
            return []

        query = """
        MATCH ()-[f:FOLLOWS]->()
        RETURN DISTINCT f.book as book
        """

        books = []
        with self._driver.session() as session:
            records = session.run(query)
            for record in records:
                books.append(record['book'])

        return books

    # ========================================================================
    # MUTATIONS
    # ========================================================================

    def add_author(self, author: Author) -> bool:
        """Add an author node to the graph."""
        if not self._connected:
            return False

        query = """
        MERGE (a:Author {name: $name})
        SET a.era = $era, a.domain = $domain
        """

        with self._driver.session() as session:
            session.run(query,
                name=author.name,
                era=author.era,
                domain=author.domain,
            )

        return True

    def add_book(self, book: Book) -> bool:
        """Add a book node and link to author."""
        if not self._connected:
            return False

        # Create book
        query_book = """
        MERGE (b:Book {id: $id})
        SET b.title = $title, b.author = $author, b.filename = $filename,
            b.genre = $genre, b.processed_at = $processed_at,
            b.n_bonds = $n_bonds, b.n_sentences = $n_sentences,
            b.n_chapters = $n_chapters
        """

        # Link to author
        query_wrote = """
        MATCH (a:Author {name: $author}), (b:Book {id: $book_id})
        MERGE (a)-[:WROTE]->(b)
        """

        with self._driver.session() as session:
            session.run(query_book,
                id=book.id,
                title=book.title,
                author=book.author,
                filename=book.filename,
                genre=book.genre,
                processed_at=book.processed_at.isoformat() if book.processed_at else None,
                n_bonds=book.n_bonds,
                n_sentences=book.n_sentences,
                n_chapters=book.n_chapters,
            )
            session.run(query_wrote,
                author=book.author,
                book_id=book.id,
            )

        return True

    def add_bond(self, bond: Bond) -> str:
        """Add a bond node to the graph. Returns bond_id."""
        if not self._connected:
            return ''

        query = """
        MERGE (b:Bond {id: $id})
        SET b.adj = $adj, b.noun = $noun,
            b.A = $A, b.S = $S, b.tau = $tau
        """

        bond_id = f"{bond.adj}_{bond.noun}" if bond.adj else bond.noun

        with self._driver.session() as session:
            session.run(query,
                id=bond_id,
                adj=bond.adj,
                noun=bond.noun,
                A=bond.A,
                S=bond.S,
                tau=bond.tau,
            )

        return bond_id

    def add_bond_to_book(self, bond_id: str, book_id: str,
                         chapter: int, sentence: int, position: int) -> bool:
        """Add CONTAINS relationship from book to bond."""
        if not self._connected:
            return False

        query = """
        MATCH (book:Book {id: $book_id}), (bond:Bond {id: $bond_id})
        MERGE (book)-[:CONTAINS {chapter: $chapter, sentence: $sentence, position: $position}]->(bond)
        """

        with self._driver.session() as session:
            session.run(query,
                book_id=book_id,
                bond_id=bond_id,
                chapter=chapter,
                sentence=sentence,
                position=position,
            )

        return True

    def add_follows(self, source: Bond, target: Bond,
                    book_id: str, chapter: int, sentence: int,
                    position: int) -> bool:
        """Add a FOLLOWS edge between bonds."""
        if not self._connected:
            return False

        source_id = f"{source.adj}_{source.noun}" if source.adj else source.noun
        target_id = f"{target.adj}_{target.noun}" if target.adj else target.noun

        query = """
        MATCH (s:Bond {id: $source_id}), (t:Bond {id: $target_id})
        MERGE (s)-[f:FOLLOWS {book_id: $book_id, chapter: $chapter,
                              sentence: $sentence, position: $position}]->(t)
        """

        with self._driver.session() as session:
            session.run(query,
                source_id=source_id,
                target_id=target_id,
                book_id=book_id,
                chapter=chapter,
                sentence=sentence,
                position=position,
            )

        return True

    def load_trajectory(self, trajectory: Trajectory, book_id: str,
                        chapter: int = 0) -> int:
        """Load a full trajectory into Neo4j."""
        if not self._connected:
            return 0

        count = 0
        for i, bond in enumerate(trajectory.bonds):
            self.add_bond(bond)
            if i > 0:
                self.add_follows(
                    trajectory.bonds[i-1], bond,
                    book_id=book_id,
                    chapter=chapter,
                    sentence=i // 5,  # Approximate
                    position=i,
                )
            count += 1

        return count

    def update_book_stats(self, book_id: str, n_bonds: int,
                          n_sentences: int, n_chapters: int) -> bool:
        """Update book statistics after processing."""
        if not self._connected:
            return False

        query = """
        MATCH (b:Book {id: $book_id})
        SET b.n_bonds = $n_bonds, b.n_sentences = $n_sentences,
            b.n_chapters = $n_chapters, b.processed_at = $processed_at
        """

        with self._driver.session() as session:
            session.run(query,
                book_id=book_id,
                n_bonds=n_bonds,
                n_sentences=n_sentences,
                n_chapters=n_chapters,
                processed_at=datetime.now().isoformat(),
            )

        return True

    def stats(self) -> Dict:
        """Get graph statistics."""
        if not self._connected:
            return {'connected': False}

        query_bonds = "MATCH (b:Bond) RETURN count(b) as count"
        query_follows = "MATCH ()-[f:FOLLOWS]->() RETURN count(f) as count"
        query_books = "MATCH (b:Book) RETURN count(b) as count"
        query_authors = "MATCH (a:Author) RETURN count(a) as count"

        with self._driver.session() as session:
            n_bonds = session.run(query_bonds).single()['count']
            n_follows = session.run(query_follows).single()['count']
            n_books = session.run(query_books).single()['count']
            n_authors = session.run(query_authors).single()['count']

        return {
            'connected': True,
            'n_bonds': n_bonds,
            'n_follows': n_follows,
            'n_books': n_books,
            'n_authors': n_authors,
        }

    def get_all_books(self) -> List[Book]:
        """Get all books in the graph."""
        if not self._connected:
            return []

        query = """
        MATCH (b:Book)
        RETURN b.id, b.title, b.author, b.filename, b.genre,
               b.processed_at, b.n_bonds, b.n_sentences, b.n_chapters
        ORDER BY b.author, b.title
        """

        books = []
        with self._driver.session() as session:
            records = session.run(query)
            for r in records:
                books.append(Book(
                    id=r['b.id'],
                    title=r['b.title'],
                    author=r['b.author'],
                    filename=r['b.filename'] or '',
                    genre=r['b.genre'] or '',
                    n_bonds=r['b.n_bonds'] or 0,
                    n_sentences=r['b.n_sentences'] or 0,
                    n_chapters=r['b.n_chapters'] or 0,
                ))

        return books

    def get_book_trajectory(self, book_id: str, chapter: Optional[int] = None,
                            limit: int = 1000) -> Trajectory:
        """Get the full trajectory of a book."""
        if not self._connected:
            return Trajectory()

        if chapter is not None:
            query = """
            MATCH (b:Book {id: $book_id})-[c:CONTAINS {chapter: $chapter}]->(bond:Bond)
            RETURN bond.adj, bond.noun, bond.A, bond.S, bond.tau, c.position
            ORDER BY c.position
            LIMIT $limit
            """
            params = {'book_id': book_id, 'chapter': chapter, 'limit': limit}
        else:
            query = """
            MATCH (b:Book {id: $book_id})-[c:CONTAINS]->(bond:Bond)
            RETURN bond.adj, bond.noun, bond.A, bond.S, bond.tau, c.chapter, c.position
            ORDER BY c.chapter, c.position
            LIMIT $limit
            """
            params = {'book_id': book_id, 'limit': limit}

        trajectory = Trajectory(metadata={'book_id': book_id})
        with self._driver.session() as session:
            records = session.run(query, **params)
            for r in records:
                trajectory.bonds.append(Bond(
                    adj=r['bond.adj'],
                    noun=r['bond.noun'],
                    A=r['bond.A'] or 0.0,
                    S=r['bond.S'] or 0.0,
                    tau=r['bond.tau'] or 2.5,
                ))

        return trajectory

    def get_author_trajectory(self, author_name: str, limit: int = 5000) -> Trajectory:
        """Get combined trajectory of all books by an author."""
        if not self._connected:
            return Trajectory()

        query = """
        MATCH (a:Author {name: $author})-[:WROTE]->(b:Book)-[c:CONTAINS]->(bond:Bond)
        RETURN bond.adj, bond.noun, bond.A, bond.S, bond.tau, b.id as book_id, c.position
        ORDER BY b.id, c.position
        LIMIT $limit
        """

        trajectory = Trajectory(metadata={'author': author_name})
        with self._driver.session() as session:
            records = session.run(query, author=author_name, limit=limit)
            for r in records:
                trajectory.bonds.append(Bond(
                    adj=r['bond.adj'],
                    noun=r['bond.noun'],
                    A=r['bond.A'] or 0.0,
                    S=r['bond.S'] or 0.0,
                    tau=r['bond.tau'] or 2.5,
                ))

        return trajectory


# ============================================================================
# SINGLETON
# ============================================================================

_neo4j_instance: Optional[Neo4jData] = None


def get_neo4j() -> Neo4jData:
    """Get the singleton Neo4jData instance."""
    global _neo4j_instance
    if _neo4j_instance is None:
        _neo4j_instance = Neo4jData()
    return _neo4j_instance
