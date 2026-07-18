from openai import OpenAI

from src.eval.retry import with_retry


def _is_retryable(e: Exception) -> bool:
    """Transient errors worth retrying: dropped connections (e.g. a local vLLM
    server briefly unreachable) and 5xx/429 status codes."""
    import openai
    if isinstance(e, openai.APIConnectionError):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code in (429, 500, 502, 503, 529)
    return False


class OpenAIClient:
    def __init__(self, base_url="http://localhost:8000/v1", model="organism", api_key="x"):
        self._c = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
    def complete(self, prompt: str, max_tokens=512, temperature=0.7, extra_body=None) -> str:
        def call():
            r = self._c.chat.completions.create(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
                **({"extra_body": extra_body} if extra_body else {}),
            )
            return r.choices[0].message.content
        return with_retry(call, retryable=_is_retryable)
