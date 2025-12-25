# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'MeaningGraph':
        from .meaning_graph import MeaningGraph
        return MeaningGraph
    elif name == 'GraphConfig':
        from .meaning_graph import GraphConfig
        return GraphConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['MeaningGraph', 'GraphConfig']
