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
    # New unified navigation system
    elif name == 'SemanticNavigator':
        from .navigator import SemanticNavigator
        return SemanticNavigator
    elif name == 'NavigationResult':
        from .navigator import NavigationResult
        return NavigationResult
    elif name == 'NavigationQuality':
        from .navigator import NavigationQuality
        return NavigationQuality
    elif name == 'NavigationGoal':
        from .navigator import NavigationGoal
        return NavigationGoal
    # Individual engines
    elif name == 'SemanticLaser':
        from .semantic_laser import SemanticLaser
        return SemanticLaser
    elif name == 'MonteCarloRenderer':
        from .monte_carlo_renderer import MonteCarloRenderer
        return MonteCarloRenderer
    elif name == 'ParadoxDetector':
        from .paradox_detector import ParadoxDetector
        return ParadoxDetector
    elif name == 'StormLogosBuilder':
        from .storm_logos import StormLogosBuilder
        return StormLogosBuilder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Core
    'Decomposer', 'TreeBuilder', 'Renderer', 'FeedbackAnalyzer',
    # Unified navigator
    'SemanticNavigator', 'NavigationResult', 'NavigationQuality', 'NavigationGoal',
    # Individual engines
    'SemanticLaser', 'MonteCarloRenderer', 'ParadoxDetector', 'StormLogosBuilder',
]
