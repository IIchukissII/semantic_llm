"""
Input Module - Processing external content into experience.

Entry point for:
- Books (files, text)
- Articles
- Conversations
- Future: API endpoints

Each source type has appropriate initial weight:
- Books: w_max (established knowledge)
- Articles: 0.8 * w_max (curated but less authoritative)
- Conversation: 2 * w_min (needs reinforcement)
"""

from .book_processor import (
    BookProcessor,
    ProcessingResult,
    SourceWeight,
    process_book,
    process_library,
)

__all__ = [
    'BookProcessor',
    'ProcessingResult',
    'SourceWeight',
    'process_book',
    'process_library',
]
