# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == 'Decomposer':
        from .decomposer import Decomposer
        return Decomposer
    elif name == 'TreeBuilder':
        from .tree_builder import TreeBuilder
        return TreeBuilder
    elif name == 'Renderer':
        from .renderer import Renderer
        return Renderer
    elif name == 'FeedbackAnalyzer':
        from .feedback import FeedbackAnalyzer
        return FeedbackAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ['Decomposer', 'TreeBuilder', 'Renderer', 'FeedbackAnalyzer']
