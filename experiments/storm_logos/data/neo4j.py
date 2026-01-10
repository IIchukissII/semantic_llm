"""Neo4j Data Layer.

Connection to Neo4j for book trajectories and FOLLOWS edges.

Extended Schema:
    (:Author {name, era, domain})
    (:Book {id, title, author, filename, genre, processed_at, n_bonds, n_sentences})
    (:Bond {id, adj, noun, A, S, tau})

    (:Author)-[:WROTE]->(:Book)
    (:Book)-[:CONTAINS {chapter, sentence, position}]->(:Bond)
    (:Bond)-[:FOLLOWS {book_id, chapter, sentence, position, weight, last_used, last_reinforced, source}]->(:Bond)

Weight Dynamics:
    dw/dt = lambda * (w_target - w)

    Decay formula:  w(t+dt) = w_min + (w - w_min) * e^(-lambda_forget * dt)
    Learn formula:  w(t+1) = w_max - (w_max - w) * e^(-lambda_learn)

    Parameters:
        w_min = 0.1         Floor - never fully forgotten
        w_max = 1.0         Ceiling - fully learned
        lambda_learn = 0.3  Learning rate (per reinforcement)
        lambda_forget = 0.05    Forgetting rate (per day)

    Weight Sources:
        Corpus (books):     1.0   - Established knowledge
        Conversation:       0.2   - Needs reinforcement
        Context-inferred:   0.1   - Weakest, most uncertain
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
    """A FOLLOWS edge between bonds with weight dynamics."""
    source_id: str
    target_id: str
    book_id: str
    chapter: int
    sentence: int
    position: int
    weight: float = 1.0       # Edge weight [0.1, 1.0]
    source: str = 'corpus'    # 'corpus', 'conversation', 'context'
    last_used: Optional[datetime] = None
    last_reinforced: Optional[datetime] = None


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
                    position: int, weight: float = 1.0,
                    edge_source: str = 'corpus') -> bool:
        """Add a FOLLOWS edge between bonds with weight dynamics.

        Args:
            source: Source bond
            target: Target bond
            book_id: Book ID
            chapter: Chapter number
            sentence: Sentence number
            position: Position in trajectory
            weight: Initial weight (default 1.0 for corpus)
            edge_source: Source type ('corpus', 'conversation', 'context')
        """
        if not self._connected:
            return False

        source_id = f"{source.adj}_{source.noun}" if source.adj else source.noun
        target_id = f"{target.adj}_{target.noun}" if target.adj else target.noun

        query = """
        MATCH (s:Bond {id: $source_id}), (t:Bond {id: $target_id})
        MERGE (s)-[f:FOLLOWS {book_id: $book_id, chapter: $chapter,
                              sentence: $sentence, position: $position}]->(t)
        ON CREATE SET
            f.weight = $weight,
            f.source = $edge_source,
            f.created_at = datetime(),
            f.last_used = datetime()
        """

        with self._driver.session() as session:
            session.run(query,
                source_id=source_id,
                target_id=target_id,
                book_id=book_id,
                chapter=chapter,
                sentence=sentence,
                position=position,
                weight=weight,
                edge_source=edge_source,
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

    # ========================================================================
    # WEIGHT DYNAMICS - LEARNING & FORGETTING
    # ========================================================================

    def apply_decay(self, days_elapsed: float = 1.0,
                    dry_run: bool = False) -> Dict:
        """
        Apply forgetting decay to user-learned FOLLOWS edge weights.

        IMPORTANT: Only decays edges from 'conversation' or 'context' sources.
        Corpus edges (from books) are permanent and never decay.

        Formula: w(t+dt) = w_min + (w - w_min) * e^(-lambda * dt)

        This implements the "nightly decay" where user-walked paths not recently
        reinforced gradually lose weight toward w_min (0.1).

        Args:
            days_elapsed: Days since last decay (default 1.0 for nightly)
            dry_run: If True, compute but don't apply changes

        Returns:
            Statistics about the decay operation
        """
        if not self._connected:
            return {"error": "Not connected"}

        from .weight_dynamics import W_MIN, LAMBDA_FORGET, DORMANCY_THRESHOLD
        import math

        decay_factor = math.exp(-LAMBDA_FORGET * days_elapsed)

        with self._driver.session() as session:
            if dry_run:
                # Preview what would happen - only user-learned edges
                result = session.run("""
                    MATCH ()-[f:FOLLOWS]->()
                    WHERE f.weight IS NOT NULL
                      AND f.weight > $w_min
                      AND (f.source IS NULL OR f.source <> 'corpus')
                    WITH f,
                         f.weight as w_before,
                         $w_min + (f.weight - $w_min) * $decay_factor as w_after
                    RETURN count(f) as edge_count,
                           sum(w_before) as total_before,
                           sum(w_after) as total_after,
                           sum(w_before - w_after) as total_decay,
                           avg(w_before) as avg_before,
                           avg(w_after) as avg_after,
                           sum(CASE WHEN w_before > $threshold AND w_after <= $threshold THEN 1 ELSE 0 END) as newly_dormant
                """, w_min=W_MIN, decay_factor=decay_factor,
                     threshold=DORMANCY_THRESHOLD)

                record = result.single()
                return {
                    "dry_run": True,
                    "days_elapsed": days_elapsed,
                    "decay_factor": decay_factor,
                    "edges_affected": record["edge_count"],
                    "total_weight_before": record["total_before"],
                    "total_weight_after": record["total_after"],
                    "total_decay": record["total_decay"],
                    "avg_weight_before": record["avg_before"],
                    "avg_weight_after": record["avg_after"],
                    "newly_dormant": record["newly_dormant"],
                    "note": "Only user-learned edges (not corpus) are decayed"
                }
            else:
                # Actually apply the decay - only user-learned edges
                result = session.run("""
                    MATCH ()-[f:FOLLOWS]->()
                    WHERE f.weight IS NOT NULL
                      AND f.weight > $w_min
                      AND (f.source IS NULL OR f.source <> 'corpus')
                    WITH f,
                         f.weight as w_before,
                         $w_min + (f.weight - $w_min) * $decay_factor as w_after
                    SET f.weight = w_after,
                        f.last_decay = datetime()
                    RETURN count(f) as edge_count,
                           sum(w_before) as total_before,
                           sum(w_after) as total_after,
                           sum(w_before - w_after) as total_decay,
                           sum(CASE WHEN w_before > $threshold AND w_after <= $threshold THEN 1 ELSE 0 END) as newly_dormant
                """, w_min=W_MIN, decay_factor=decay_factor,
                     threshold=DORMANCY_THRESHOLD)

                record = result.single()
                return {
                    "dry_run": False,
                    "days_elapsed": days_elapsed,
                    "decay_factor": decay_factor,
                    "edges_affected": record["edge_count"],
                    "total_weight_before": record["total_before"],
                    "total_weight_after": record["total_after"],
                    "total_decay": record["total_decay"],
                    "newly_dormant": record["newly_dormant"],
                    "applied_at": datetime.now().isoformat(),
                    "note": "Only user-learned edges (not corpus) were decayed"
                }

    def apply_decay_since_last_use(self, dry_run: bool = False) -> Dict:
        """
        Apply decay based on each edge's individual last_used timestamp.

        IMPORTANT: Only decays edges from 'conversation' or 'context' sources.
        Corpus edges (from books) are permanent and never decay.

        More accurate than apply_decay() - decays each edge based on
        how long since it was actually used.

        Args:
            dry_run: If True, compute but don't apply changes

        Returns:
            Statistics about the decay operation
        """
        if not self._connected:
            return {"error": "Not connected"}

        from .weight_dynamics import W_MIN, LAMBDA_FORGET, DORMANCY_THRESHOLD

        with self._driver.session() as session:
            if dry_run:
                result = session.run("""
                    MATCH ()-[f:FOLLOWS]->()
                    WHERE f.weight IS NOT NULL
                      AND f.weight > $w_min
                      AND f.last_used IS NOT NULL
                      AND (f.source IS NULL OR f.source <> 'corpus')
                    WITH f,
                         f.weight as w_before,
                         duration.inDays(f.last_used, datetime()).days as days_since,
                         $w_min + (f.weight - $w_min) * exp(-$lambda * duration.inDays(f.last_used, datetime()).days) as w_after
                    RETURN count(f) as edge_count,
                           sum(w_before) as total_before,
                           sum(w_after) as total_after,
                           avg(days_since) as avg_days_since_use,
                           max(days_since) as max_days_since_use,
                           sum(CASE WHEN w_before > $threshold AND w_after <= $threshold THEN 1 ELSE 0 END) as newly_dormant
                """, w_min=W_MIN, lambda_val=LAMBDA_FORGET,
                     threshold=DORMANCY_THRESHOLD)

                record = result.single()
                if record["edge_count"] == 0:
                    return {
                        "dry_run": True,
                        "edges_with_timestamp": 0,
                        "note": "No user-learned edges have last_used timestamp set"
                    }

                return {
                    "dry_run": True,
                    "edges_affected": record["edge_count"],
                    "total_weight_before": record["total_before"],
                    "total_weight_after": record["total_after"],
                    "avg_days_since_use": record["avg_days_since_use"],
                    "max_days_since_use": record["max_days_since_use"],
                    "newly_dormant": record["newly_dormant"],
                    "note": "Only user-learned edges (not corpus) are decayed"
                }
            else:
                # Apply decay based on individual timestamps - only user-learned edges
                result = session.run("""
                    MATCH ()-[f:FOLLOWS]->()
                    WHERE f.weight IS NOT NULL
                      AND f.weight > $w_min
                      AND f.last_used IS NOT NULL
                      AND (f.source IS NULL OR f.source <> 'corpus')
                    WITH f,
                         f.weight as w_before,
                         duration.inDays(f.last_used, datetime()).days as days_since
                    WITH f, w_before, days_since,
                         $w_min + (w_before - $w_min) * exp(-$lambda * days_since) as w_after
                    SET f.weight = w_after,
                        f.last_decay = datetime()
                    RETURN count(f) as edge_count,
                           sum(w_before) as total_before,
                           sum(w_after) as total_after,
                           sum(CASE WHEN w_before > $threshold AND w_after <= $threshold THEN 1 ELSE 0 END) as newly_dormant
                """, w_min=W_MIN, lambda_val=LAMBDA_FORGET,
                     threshold=DORMANCY_THRESHOLD)

                record = result.single()
                return {
                    "dry_run": False,
                    "edges_affected": record["edge_count"],
                    "total_weight_before": record["total_before"],
                    "total_weight_after": record["total_after"],
                    "newly_dormant": record["newly_dormant"],
                    "applied_at": datetime.now().isoformat(),
                    "note": "Only user-learned edges (not corpus) were decayed"
                }

    def reinforce_edge(self, source_id: str, target_id: str) -> bool:
        """
        Reinforce a FOLLOWS edge (increase weight via learning).

        Call this when a transition is walked during navigation.
        Formula: w += 0.05 (up to w_max)

        Args:
            source_id: Source bond ID
            target_id: Target bond ID

        Returns:
            True if edge was reinforced
        """
        if not self._connected:
            return False

        from .weight_dynamics import W_MAX, LEARNING_INCREMENT

        query = """
        MATCH (s:Bond {id: $source_id})-[f:FOLLOWS]->(t:Bond {id: $target_id})
        SET f.weight = CASE
            WHEN f.weight IS NULL THEN $increment
            WHEN f.weight < $w_max THEN f.weight + $increment
            ELSE f.weight
        END,
        f.last_used = datetime(),
        f.last_reinforced = datetime()
        RETURN f.weight as new_weight
        """

        with self._driver.session() as session:
            result = session.run(query,
                source_id=source_id,
                target_id=target_id,
                w_max=W_MAX,
                increment=LEARNING_INCREMENT)
            record = result.single()
            return record is not None

    def reinforce_transition(self, source: Bond, target: Bond) -> bool:
        """
        Reinforce a transition between two bonds.

        Convenience wrapper for reinforce_edge.

        Args:
            source: Source bond
            target: Target bond

        Returns:
            True if edge was reinforced
        """
        source_id = f"{source.adj}_{source.noun}" if source.adj else source.noun
        target_id = f"{target.adj}_{target.noun}" if target.adj else target.noun
        return self.reinforce_edge(source_id, target_id)

    def mark_edge_used(self, source_id: str, target_id: str):
        """
        Mark a FOLLOWS edge as recently used (updates last_used timestamp).

        Call this during navigation to track edge usage for decay.

        Args:
            source_id: Source bond ID
            target_id: Target bond ID
        """
        if not self._connected:
            return

        query = """
        MATCH (s:Bond {id: $source_id})-[f:FOLLOWS]->(t:Bond {id: $target_id})
        SET f.last_used = datetime()
        """

        with self._driver.session() as session:
            session.run(query, source_id=source_id, target_id=target_id)

    def get_decay_stats(self) -> Dict:
        """
        Get statistics about edge weights and decay state.

        Returns:
            Dictionary with weight distribution and dormancy stats
        """
        if not self._connected:
            return {"error": "Not connected"}

        from .weight_dynamics import DORMANCY_THRESHOLD

        with self._driver.session() as session:
            result = session.run("""
                MATCH ()-[f:FOLLOWS]->()
                RETURN count(f) as total_edges,
                       avg(coalesce(f.weight, 1.0)) as avg_weight,
                       min(coalesce(f.weight, 1.0)) as min_weight,
                       max(coalesce(f.weight, 1.0)) as max_weight,
                       sum(CASE WHEN coalesce(f.weight, 1.0) <= $threshold THEN 1 ELSE 0 END) as dormant_count,
                       sum(CASE WHEN coalesce(f.weight, 1.0) > $threshold THEN 1 ELSE 0 END) as active_count,
                       sum(CASE WHEN coalesce(f.weight, 1.0) >= 0.9 THEN 1 ELSE 0 END) as saturated_count,
                       sum(CASE WHEN f.last_used IS NOT NULL THEN 1 ELSE 0 END) as edges_with_timestamp,
                       sum(CASE WHEN f.weight IS NOT NULL THEN 1 ELSE 0 END) as edges_with_weight
            """, threshold=DORMANCY_THRESHOLD)

            record = result.single()
            total = record["total_edges"]

            return {
                "total_edges": total,
                "avg_weight": record["avg_weight"],
                "min_weight": record["min_weight"],
                "max_weight": record["max_weight"],
                "dormant_count": record["dormant_count"],
                "active_count": record["active_count"],
                "saturated_count": record["saturated_count"],
                "dormant_percentage": (record["dormant_count"] / total * 100) if total > 0 else 0,
                "edges_with_timestamp": record["edges_with_timestamp"],
                "edges_with_weight": record["edges_with_weight"],
                "dormancy_threshold": DORMANCY_THRESHOLD
            }

    def initialize_weights(self, default_weight: float = 1.0) -> Dict:
        """
        Initialize weights on edges that don't have them.

        Useful when adding weight support to existing edges.

        Args:
            default_weight: Weight to set (default 1.0 for corpus edges)

        Returns:
            Statistics about initialization
        """
        if not self._connected:
            return {"error": "Not connected"}

        with self._driver.session() as session:
            result = session.run("""
                MATCH ()-[f:FOLLOWS]->()
                WHERE f.weight IS NULL
                SET f.weight = $weight,
                    f.source = 'corpus',
                    f.last_used = datetime()
                RETURN count(f) as initialized
            """, weight=default_weight)

            record = result.single()
            return {
                "initialized": record["initialized"],
                "default_weight": default_weight
            }

    def get_weight_distribution(self, buckets: int = 10) -> List[Dict]:
        """
        Get histogram of edge weights.

        Args:
            buckets: Number of histogram buckets

        Returns:
            List of {range_start, range_end, count} dicts
        """
        if not self._connected:
            return []

        from .weight_dynamics import W_MIN, W_MAX

        bucket_size = (W_MAX - W_MIN) / buckets
        distribution = []

        with self._driver.session() as session:
            for i in range(buckets):
                low = W_MIN + i * bucket_size
                high = low + bucket_size

                result = session.run("""
                    MATCH ()-[f:FOLLOWS]->()
                    WHERE coalesce(f.weight, 1.0) >= $low AND coalesce(f.weight, 1.0) < $high
                    RETURN count(f) as count
                """, low=low, high=high)

                record = result.single()
                distribution.append({
                    "range_start": round(low, 2),
                    "range_end": round(high, 2),
                    "count": record["count"]
                })

        return distribution

    # ========================================================================
    # LEARNING: Runtime Bond Learning
    # ========================================================================

    def learn_bond(self, bond: Bond, source: str = 'conversation') -> str:
        """Learn a new bond from conversation (creates or updates node).

        Unlike add_bond() which is for corpus loading, this method:
        - Sets appropriate source marker
        - Is idempotent (safe to call multiple times)

        Args:
            bond: Bond to learn
            source: Source type ('conversation', 'context')

        Returns:
            Bond ID
        """
        if not self._connected:
            return ''

        bond_id = f"{bond.adj}_{bond.noun}" if bond.adj else bond.noun

        query = """
        MERGE (b:Bond {id: $id})
        ON CREATE SET
            b.adj = $adj,
            b.noun = $noun,
            b.A = $A,
            b.S = $S,
            b.tau = $tau,
            b.source = $source,
            b.created_at = datetime()
        ON MATCH SET
            b.last_used = datetime()
        RETURN b.id
        """

        with self._driver.session() as session:
            session.run(query,
                id=bond_id,
                adj=bond.adj,
                noun=bond.noun,
                A=bond.A,
                S=bond.S,
                tau=bond.tau,
                source=source,
            )

        return bond_id

    def learn_transition(self, source: Bond, target: Bond,
                         conversation_id: str = 'default',
                         source_type: str = 'conversation') -> bool:
        """Learn a transition between bonds from conversation.

        Creates FOLLOWS edge with conversation weight (0.2) that is
        subject to decay. Unlike add_follows() which creates corpus edges.

        Args:
            source: Source bond
            target: Target bond
            conversation_id: Conversation identifier
            source_type: 'conversation' or 'context'

        Returns:
            True if edge was created/updated
        """
        if not self._connected:
            return False

        from .weight_dynamics import WEIGHT_CONVERSATION, WEIGHT_CONTEXT

        source_id = f"{source.adj}_{source.noun}" if source.adj else source.noun
        target_id = f"{target.adj}_{target.noun}" if target.adj else target.noun

        # Determine initial weight based on source type
        init_weight = WEIGHT_CONVERSATION if source_type == 'conversation' else WEIGHT_CONTEXT

        query = """
        MATCH (s:Bond {id: $source_id}), (t:Bond {id: $target_id})
        MERGE (s)-[f:FOLLOWS {conversation_id: $conv_id}]->(t)
        ON CREATE SET
            f.weight = $init_weight,
            f.source = $source_type,
            f.created_at = datetime(),
            f.last_used = datetime()
        ON MATCH SET
            f.weight = CASE
                WHEN f.weight < 1.0 THEN f.weight + 0.05
                ELSE f.weight
            END,
            f.last_used = datetime()
        RETURN f.weight as weight
        """

        with self._driver.session() as session:
            result = session.run(query,
                source_id=source_id,
                target_id=target_id,
                conv_id=conversation_id,
                init_weight=init_weight,
                source_type=source_type,
            )
            record = result.single()
            return record is not None

    def learn_trajectory(self, bonds: List[Bond],
                         conversation_id: str = 'default',
                         source_type: str = 'conversation') -> int:
        """Learn a sequence of bonds from conversation.

        Creates Bond nodes and FOLLOWS edges for the trajectory.

        Args:
            bonds: List of bonds in order
            conversation_id: Conversation identifier
            source_type: 'conversation' or 'context'

        Returns:
            Number of edges created
        """
        if not self._connected or len(bonds) < 2:
            return 0

        count = 0
        for i in range(len(bonds)):
            # Create/update bond node
            self.learn_bond(bonds[i], source=source_type)

            # Create transition to next bond
            if i > 0:
                self.learn_transition(
                    bonds[i-1], bonds[i],
                    conversation_id=conversation_id,
                    source_type=source_type
                )
                count += 1

        return count

    def get_learned_bonds(self, limit: int = 100) -> List[Bond]:
        """Get bonds learned from conversations (not corpus).

        Args:
            limit: Maximum bonds to return

        Returns:
            List of learned bonds
        """
        if not self._connected:
            return []

        query = """
        MATCH (b:Bond)
        WHERE b.source IN ['conversation', 'context']
        RETURN b.adj, b.noun, b.A, b.S, b.tau
        ORDER BY b.created_at DESC
        LIMIT $limit
        """

        bonds = []
        with self._driver.session() as session:
            records = session.run(query, limit=limit)
            for r in records:
                bonds.append(Bond(
                    adj=r['b.adj'],
                    noun=r['b.noun'],
                    A=r['b.A'] or 0.0,
                    S=r['b.S'] or 0.0,
                    tau=r['b.tau'] or 2.5,
                ))

        return bonds

    def get_conversation_trajectory(self, conversation_id: str,
                                    limit: int = 100) -> Trajectory:
        """Get the trajectory of a specific conversation.

        Args:
            conversation_id: Conversation identifier
            limit: Maximum bonds to return

        Returns:
            Trajectory of the conversation
        """
        if not self._connected:
            return Trajectory()

        query = """
        MATCH (b1:Bond)-[f:FOLLOWS {conversation_id: $conv_id}]->(b2:Bond)
        RETURN b1.adj, b1.noun, b1.A, b1.S, b1.tau,
               b2.adj, b2.noun, b2.A, b2.S, b2.tau,
               f.created_at
        ORDER BY f.created_at
        LIMIT $limit
        """

        trajectory = Trajectory(metadata={'conversation_id': conversation_id})
        seen = set()

        with self._driver.session() as session:
            records = session.run(query, conv_id=conversation_id, limit=limit)
            for r in records:
                # Add source bond if not seen
                bond1_id = f"{r['b1.adj']}_{r['b1.noun']}"
                if bond1_id not in seen:
                    trajectory.bonds.append(Bond(
                        adj=r['b1.adj'],
                        noun=r['b1.noun'],
                        A=r['b1.A'] or 0.0,
                        S=r['b1.S'] or 0.0,
                        tau=r['b1.tau'] or 2.5,
                    ))
                    seen.add(bond1_id)

                # Add target bond if not seen
                bond2_id = f"{r['b2.adj']}_{r['b2.noun']}"
                if bond2_id not in seen:
                    trajectory.bonds.append(Bond(
                        adj=r['b2.adj'],
                        noun=r['b2.noun'],
                        A=r['b2.A'] or 0.0,
                        S=r['b2.S'] or 0.0,
                        tau=r['b2.tau'] or 2.5,
                    ))
                    seen.add(bond2_id)

        return trajectory

    def get_learning_stats(self) -> Dict:
        """Get statistics about learned (non-corpus) content.

        Returns:
            Dictionary with learning statistics
        """
        if not self._connected:
            return {"error": "Not connected"}

        with self._driver.session() as session:
            # Count learned bonds
            result = session.run("""
                MATCH (b:Bond)
                WHERE b.source IN ['conversation', 'context']
                RETURN count(b) as count
            """)
            n_learned_bonds = result.single()["count"]

            # Count learned edges
            result = session.run("""
                MATCH ()-[f:FOLLOWS]->()
                WHERE f.source IN ['conversation', 'context']
                RETURN count(f) as count,
                       avg(f.weight) as avg_weight
            """)
            record = result.single()
            n_learned_edges = record["count"]
            avg_learned_weight = record["avg_weight"] or 0.0

            # Count unique conversations
            result = session.run("""
                MATCH ()-[f:FOLLOWS]->()
                WHERE f.conversation_id IS NOT NULL
                RETURN count(DISTINCT f.conversation_id) as count
            """)
            n_conversations = result.single()["count"]

            return {
                "learned_bonds": n_learned_bonds,
                "learned_edges": n_learned_edges,
                "avg_learned_weight": avg_learned_weight,
                "conversations": n_conversations,
            }

    # ========================================================================
    # SYNC HELPERS
    # ========================================================================

    def get_existing_bond_ids(self) -> set:
        """Get all existing bond IDs in Neo4j.

        Useful for sync operations to check what's already present.

        Returns:
            Set of bond IDs (format: "{adj}_{noun}")
        """
        if not self._connected:
            return set()

        query = "MATCH (b:Bond) RETURN b.id as id"

        bond_ids = set()
        with self._driver.session() as session:
            records = session.run(query)
            for record in records:
                bond_ids.add(record["id"])

        return bond_ids

    def bulk_add_bonds(self, bonds: List[Bond]) -> int:
        """Add multiple bonds in a single transaction.

        More efficient than calling add_bond() multiple times.

        Args:
            bonds: List of bonds to add

        Returns:
            Number of bonds added
        """
        if not self._connected or not bonds:
            return 0

        query = """
        UNWIND $bonds as bond
        MERGE (b:Bond {id: bond.id})
        SET b.adj = bond.adj, b.noun = bond.noun,
            b.A = bond.A, b.S = bond.S, b.tau = bond.tau,
            b.synced_at = datetime()
        """

        bond_data = []
        for bond in bonds:
            bond_id = f"{bond.adj}_{bond.noun}" if bond.adj else bond.noun
            bond_data.append({
                'id': bond_id,
                'adj': bond.adj,
                'noun': bond.noun,
                'A': bond.A,
                'S': bond.S,
                'tau': bond.tau,
            })

        with self._driver.session() as session:
            session.run(query, bonds=bond_data)

        return len(bond_data)

    def bulk_add_follows(self, edges: List[tuple]) -> int:
        """Add multiple FOLLOWS edges in a single transaction.

        Args:
            edges: List of (source_bond, target_bond, metadata) tuples
                   metadata should contain: book_id, chapter, sentence, position,
                   weight, source

        Returns:
            Number of edges added
        """
        if not self._connected or not edges:
            return 0

        query = """
        UNWIND $edges as edge
        MATCH (s:Bond {id: edge.source_id}), (t:Bond {id: edge.target_id})
        MERGE (s)-[f:FOLLOWS {book_id: edge.book_id}]->(t)
        ON CREATE SET
            f.chapter = edge.chapter,
            f.sentence = edge.sentence,
            f.position = edge.position,
            f.weight = edge.weight,
            f.source = edge.source,
            f.created_at = datetime(),
            f.last_used = datetime()
        """

        edge_data = []
        for source, target, meta in edges:
            source_id = f"{source.adj}_{source.noun}" if source.adj else source.noun
            target_id = f"{target.adj}_{target.noun}" if target.adj else target.noun
            edge_data.append({
                'source_id': source_id,
                'target_id': target_id,
                'book_id': meta.get('book_id', 'conversation'),
                'chapter': meta.get('chapter', 0),
                'sentence': meta.get('sentence', 0),
                'position': meta.get('position', 0),
                'weight': meta.get('weight', 0.2),
                'source': meta.get('source', 'conversation'),
            })

        with self._driver.session() as session:
            session.run(query, edges=edge_data)

        return len(edge_data)


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
