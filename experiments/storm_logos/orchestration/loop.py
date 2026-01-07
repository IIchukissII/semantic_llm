"""Main Loop: The core measure -> adapt -> generate cycle."""

from typing import Optional, List, Callable

from ..data.models import SemanticState, Metrics, Parameters, Bond
from ..metrics.engine import MetricsEngine
from ..feedback.engine import FeedbackEngine
from ..controller.engine import AdaptiveController
from ..generation.engine import GenerationEngine
from .session import Session


class MainLoop:
    """Main orchestration loop.

    The heart of Storm-Logos:
        1. Receive input
        2. MEASURE: Compute metrics
        3. FEEDBACK: Compute errors
        4. ADAPT: Adjust parameters
        5. GENERATE: Produce output
        6. Update state
        7. Repeat
    """

    def __init__(self,
                 metrics: Optional[MetricsEngine] = None,
                 feedback: Optional[FeedbackEngine] = None,
                 controller: Optional[AdaptiveController] = None,
                 generation: Optional[GenerationEngine] = None):
        self.metrics = metrics or MetricsEngine()
        self.feedback = feedback or FeedbackEngine()
        self.controller = controller or AdaptiveController()
        self.generation = generation or GenerationEngine()

        self.session = Session()
        self._callbacks: List[Callable] = []

    def step(self, input_text: str = None,
             input_state: SemanticState = None) -> dict:
        """Execute one step of the main loop.

        Args:
            input_text: Text input (for therapy mode)
            input_state: State input (for generation mode)

        Returns:
            Dictionary with step results
        """
        result = {}

        # 1. MEASURE
        if input_text:
            metrics = self.metrics.measure(text=input_text)
        elif input_state:
            metrics = self.metrics.measure(state=input_state)
        else:
            metrics = self.metrics.measure(
                trajectory=self.generation.get_trajectory()
            )

        self.session.add_metrics(metrics)
        result['metrics'] = metrics.as_dict()

        # 2. FEEDBACK
        errors = self.feedback.compute_errors(metrics)
        result['errors'] = errors.as_dict()

        # 3. ADAPT
        new_params = self.controller.adapt(errors)
        self.session.update_parameters(new_params)
        result['parameters'] = new_params.as_dict()

        # 4. GENERATE
        gen_result = self.generation.generate_next(new_params)
        result['bond'] = gen_result.bond.text
        result['new_state'] = {
            'A': gen_result.new_state.A,
            'S': gen_result.new_state.S,
            'tau': gen_result.new_state.tau,
        }

        # 5. UPDATE STATE
        self.session.update_state(gen_result.new_state)

        # 6. CALLBACKS
        for callback in self._callbacks:
            callback(result)

        return result

    def run(self, n_steps: int = 10,
            seed_state: Optional[SemanticState] = None) -> List[dict]:
        """Run multiple steps.

        Args:
            n_steps: Number of steps
            seed_state: Starting state

        Returns:
            List of step results
        """
        if seed_state:
            self.generation.reset(seed_state)
            self.session.Q = seed_state

        results = []
        for _ in range(n_steps):
            result = self.step()
            results.append(result)

        return results

    def run_until(self, condition: Callable[[dict], bool],
                  max_steps: int = 100) -> List[dict]:
        """Run until condition is met.

        Args:
            condition: Function that returns True to stop
            max_steps: Maximum steps

        Returns:
            List of step results
        """
        results = []
        for _ in range(max_steps):
            result = self.step()
            results.append(result)

            if condition(result):
                break

        return results

    def reset(self, context: str = 'default'):
        """Reset the loop.

        Args:
            context: New context for adaptation rules
        """
        self.session.reset()
        self.controller.set_context(context)
        self.controller.reset()
        self.feedback.reset()
        self.generation.reset()

    def add_callback(self, callback: Callable[[dict], None]):
        """Add callback to be called after each step."""
        self._callbacks.append(callback)

    def get_state(self) -> dict:
        """Get full current state."""
        return {
            'session': self.session.to_dict(),
            'controller_history': self.controller.get_history(5),
            'feedback_integrals': self.feedback.get_integral(),
        }
