# Lazy imports
def __getattr__(name):
    if name == 'BookProcessor':
        from .book_processor import BookProcessor
        return BookProcessor
    elif name == 'process_book':
        from .book_processor import process_book
        return process_book
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['BookProcessor', 'process_book']
