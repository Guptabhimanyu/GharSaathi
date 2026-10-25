"""Free-tier LLM router with automatic fallback.

Every provider below speaks the OpenAI-compatible chat API, so one client class covers all of them.
If a provider hits its rate/token limit, errors, or has removed the model, we move to the next one
and put the failed provider on cooldown so we don't keep hammering it.

Free catalogs change often: override any model with the *_MODEL env vars instead of editing code.
Set LLM_ORDER="groq,gemini,..." to change priority.
"""
import logging
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import (APIConnectionError, APITimeoutError, AuthenticationError,
                    NotFoundError, OpenAI, OpenAIError, RateLimitError)

load_dotenv()  # reads keys from a local .env file if present
log = logging.getLogger("llm")


class LLMUnavailable(RuntimeError):
    """Raised when every configured provider has failed or is cooling down."""

LAST_PROVIDER = ""  # which provider answered the most recent call (for the UI)


@dataclass
class Provider:
    name: str
    base_url: str
    key_env: str
    model_env: str
    default_model: str

    @property
    def model(self):
        return os.getenv(self.model_env, self.default_model)

    @property
    def enabled(self):
        return bool(os.getenv(self.key_env)) or self.name == "ollama" and bool(os.getenv("USE_OLLAMA"))


PROVIDERS = {p.name: p for p in [
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/",
             "GEMINI_API_KEY", "GEMINI_MODEL", "gemini-2.5-flash"),
    Provider("groq", "https://api.groq.com/openai/v1",
             "GROQ_API_KEY", "GROQ_MODEL", "llama-3.3-70b-versatile"),
    Provider("cerebras", "https://api.cerebras.ai/v1",
             "CEREBRAS_API_KEY", "CEREBRAS_MODEL", "gpt-oss-120b"),
    Provider("mistral", "https://api.mistral.ai/v1",
             "MISTRAL_API_KEY", "MISTRAL_MODEL", "mistral-small-latest"),
    Provider("openrouter", "https://openrouter.ai/api/v1",
             "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    Provider("ollama", "http://localhost:11434/v1",
             "OLLAMA_API_KEY", "OLLAMA_MODEL", "qwen2.5:7b"),  # local, last resort, never rate-limited
]}
DEFAULT_ORDER = ["gemini", "groq", "cerebras", "mistral", "openrouter", "ollama"]
_cooldown: dict[str, float] = {}  # provider name -> timestamp when it may be tried again


def _order():
    names = os.getenv("LLM_ORDER", ",".join(DEFAULT_ORDER)).split(",")
    return [PROVIDERS[n.strip()] for n in names if n.strip() in PROVIDERS]


def _cool_for(err: Exception) -> int:
    if isinstance(err, RateLimitError):
        return 90          # out of requests/tokens: wait a bit
    if isinstance(err, (AuthenticationError, NotFoundError)):
        return 3600        # bad key or model removed: don't retry this session
    return 30              # timeouts / 5xx


def complete(system: str, messages: list[dict], max_tokens: int = 1200) -> str:
    global LAST_PROVIDER
    errors = []
    candidates = [p for p in _order() if p.enabled]
    if not candidates:
        raise LLMUnavailable("No LLM provider configured. Set at least one of: "
                             + ", ".join(p.key_env for p in PROVIDERS.values()))
    for p in candidates:
        if _cooldown.get(p.name, 0) > time.time():
            continue
        try:
            client = OpenAI(base_url=p.base_url, api_key=os.getenv(p.key_env, "ollama"), timeout=60, max_retries=0)
            r = client.chat.completions.create(
                model=p.model, max_tokens=max_tokens, temperature=0.2,
                messages=[{"role": "system", "content": system}] + messages)
            text = (r.choices[0].message.content or "").strip()
            if not text:
                raise OpenAIError("empty response")
            LAST_PROVIDER = f"{p.name}:{p.model}"
            return text
        except OpenAIError as e:  # covers rate limit, auth, not found, timeout, connection, 5xx
            _cooldown[p.name] = time.time() + _cool_for(e)
            errors.append(f"{p.name}: {type(e).__name__}")
            log.warning("Provider %s failed (%s); falling back", p.name, type(e).__name__)
    LAST_PROVIDER = "none (showing document passages only)"
    raise LLMUnavailable("All LLM providers failed or are cooling down: " + "; ".join(errors or ["all on cooldown"]))
