"""Renderer: Convert bond skeletons to text via LLM."""

from typing import List, Optional
import requests
import json

from ..data.models import Bond


class Renderer:
    """Render bond skeletons to text using LLM.

    Supports:
    - Ollama (Mistral, Llama)
    - Direct API calls
    """

    def __init__(self,
                 model: str = 'mistral:7b',
                 base_url: str = 'http://localhost:11434'):
        self.model = model
        self.base_url = base_url

    def render(self, skeleton: List[List[Bond]],
               genre: str = 'literary',
               max_tokens: int = 200) -> str:
        """Render skeleton to text.

        Args:
            skeleton: List of sentences, each a list of bonds
            genre: Genre hint for style
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text
        """
        # Build prompt from skeleton
        prompt = self._build_prompt(skeleton, genre)

        # Call LLM
        response = self._call_llm(prompt, max_tokens)

        return response

    def _build_prompt(self, skeleton: List[List[Bond]], genre: str) -> str:
        """Build prompt from skeleton."""
        # Convert bonds to phrase hints
        hints = []
        for sent_idx, sentence in enumerate(skeleton):
            bond_texts = [b.text for b in sentence]
            hints.append(f"Sentence {sent_idx + 1}: {', '.join(bond_texts)}")

        skeleton_str = '\n'.join(hints)

        prompt = f"""Write a {genre} paragraph using these semantic elements:

{skeleton_str}

Requirements:
- Use each phrase naturally in the text
- Create coherent, flowing prose
- Match the {genre} tone and style
- Keep it concise (2-4 sentences)

Generated text:"""

        return prompt

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Call LLM via Ollama API."""
        try:
            response = requests.post(
                f'{self.base_url}/api/generate',
                json={
                    'model': self.model,
                    'prompt': prompt,
                    'stream': False,
                    'options': {
                        'num_predict': max_tokens,
                        'temperature': 0.7,
                    }
                },
                timeout=60,
            )

            if response.status_code == 200:
                return response.json().get('response', '')
            else:
                return f"[LLM Error: {response.status_code}]"

        except Exception as e:
            return f"[LLM Error: {e}]"

    def render_therapeutic(self, skeleton: List[Bond],
                           context: str = '',
                           max_tokens: int = 100) -> str:
        """Render therapeutic response.

        Args:
            skeleton: Bonds for response
            context: Therapeutic context
            max_tokens: Maximum tokens

        Returns:
            Therapeutic response
        """
        bond_texts = [b.text for b in skeleton]

        prompt = f"""You are a skilled therapist. Generate a brief therapeutic response.

Semantic elements to incorporate: {', '.join(bond_texts)}

Context: {context}

Requirements:
- Be present and authentic
- Avoid therapy-speak clichés
- Stay grounded and direct
- Maximum 2 sentences

Response:"""

        return self._call_llm(prompt, max_tokens)


class MockRenderer(Renderer):
    """Mock renderer for testing (no LLM required)."""

    def _call_llm(self, prompt: str, max_tokens: int) -> str:
        """Return mock response."""
        return "[Mock response based on semantic skeleton]"
