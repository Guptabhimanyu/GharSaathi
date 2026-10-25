"""Free-tier LLM router with automatic fallback.

Every provider speaks the OpenAI-compatible chat API. Each provider has an ordered list of models: if one model
is retired (404), rate-limited (429) or errors, we try the provider's next model, then the next provider.
Failed (provider, model) pairs go on cooldown so we don't keep hammering them.

Free catalogs change often (several defaults below were retired or replaced in 2026). Override any provider's
model list with an env var, comma-separated, e.g. GROQ_MODEL="openai/gpt-oss-120b,qwen/qwen3.6-27b".
Set LLM_ORDER="groq,gemini,..." to change provider priority.
"""
import logging
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import AuthenticationError, NotFoundError, OpenAI, OpenAIError, RateLimitError

load_dotenv()  # reads keys from a local .env file if present
log = logging.getLogger("llm")


class LLMUnavailable(RuntimeError):
    """Raised when every configured provider/model has failed or is cooling down."""


LAST_PROVIDER = ""  # which provider:model answered the most recent call (for the UI)


@dataclass
class Provider:
    name: str
    base_url: str
    key_env: str
    model_env: str
    default_models: tuple

    @property
    def models(self) -> list:
        env = os.getenv(self.model_env)
        return [m.strip() for m in env.split(",") if m.strip()] if env else list(self.default_models)

    @property
    def enabled(self) -> bool:
        return bool(os.getenv(self.key_env)) or (self.name == "ollama" and bool(os.getenv("USE_OLLAMA")))


PROVIDERS = {p.name: p for p in [
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY", "GEMINI_MODEL",
             ("gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite")),
    Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "GROQ_MODEL",
             ("openai/gpt-oss-120b", "qwen/qwen3.6-27b", "openai/gpt-oss-20b")),
    Provider("cerebras", "https://api.cerebras.ai/v1", "CEREBRAS_API_KEY", "CEREBRAS_MODEL", ("gpt-oss-120b",)),
    Provider("mistral", "https://api.mistral.ai/v1", "MISTRAL_API_KEY", "MISTRAL_MODEL", ("mistral-small-latest",)),
    Provider("openrouter", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "OPENROUTER_MODEL",
             ("openai/gpt-oss-120b:free", "meta-llama/llama-3.3-70b-instruct:free")),
    Provider("ollama", "http://localhost:11434/v1", "OLLAMA_API_KEY", "OLLAMA_MODEL", ("qwen2.5:7b",)),
]}
DEFAULT_ORDER = ["gemini", "groq", "cerebras", "mistral", "openrouter", "ollama"]
_cooldown: dict = {}  # "provider:model" or "provider:*" -> timestamp when it may be tried again


def _order():
    names = os.getenv("LLM_ORDER", ",".join(DEFAULT_ORDER)).split(",")
    return [PROVIDERS[n.strip()] for n in names if n.strip() in PROVIDERS]


def _cool_for(err: Exception) -> float:
    if isinstance(err, RateLimitError):  # honour the server's Retry-After when given (usually a few seconds)
        try:
            wait = float(err.response.headers.get("retry-after", ""))
        except Exception:
            wait = 60.0
        return min(max(wait, 5.0), 90.0)
    if isinstance(err, (AuthenticationError, NotFoundError)):
        return 3600.0      # bad key or model retired: don't retry this session
    return 30.0            # timeouts / 5xx / empty answer


def _reasoning(model: str) -> bool:
    """Reasoning models spend part of max_tokens on hidden thinking, so give them more room."""
    return any(k in model for k in ("gemini-3", "gemini-2.5", "gpt-oss", "qwen3"))


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _attempt(candidates, system, messages, max_tokens, errors):
    """One pass over every provider/model that is not cooling down. Returns text or None."""
    global LAST_PROVIDER
    for p in candidates:
        if _cooldown.get(f"{p.name}:*", 0) > time.time():
            continue
        client = OpenAI(base_url=p.base_url, api_key=os.getenv(p.key_env, "ollama"), timeout=60, max_retries=0)
        for model in p.models:
            key = f"{p.name}:{model}"
            if _cooldown.get(key, 0) > time.time():
                continue
            try:
                r = client.chat.completions.create(
                    model=model, max_tokens=max_tokens * 3 if _reasoning(model) else max_tokens, temperature=0.2,
                    messages=[{"role": "system", "content": system}] + messages)
                text = _clean(r.choices[0].message.content)
                if not text:
                    raise OpenAIError("empty response")
                LAST_PROVIDER = key
                return text
            except OpenAIError as e:  # rate limit, auth, not found, timeout, connection, 5xx, empty
                _cooldown[key] = time.time() + _cool_for(e)
                code = getattr(e, "status_code", None)
                errors.append(f"{p.name}/{model}: {type(e).__name__}{f' {code}' if code else ''}")
                log.warning("LLM %s/%s failed (%s); trying next", p.name, model, type(e).__name__)
                if isinstance(e, AuthenticationError):  # the key itself is bad: skip this provider's other models
                    _cooldown[f"{p.name}:*"] = time.time() + 3600
                    break
    return None


def complete(system: str, messages: list, max_tokens: int = 1200) -> str:
    global LAST_PROVIDER
    candidates = [p for p in _order() if p.enabled]
    if not candidates:
        raise LLMUnavailable("No LLM key found. Add at least one of: "
                             + ", ".join(p.key_env for p in PROVIDERS.values()))
    names, errors = {p.name for p in candidates}, []
    for attempt in (1, 2):
        text = _attempt(candidates, system, messages, max_tokens, errors)
        if text:
            return text
        if attempt == 1:  # everything is rate-limited for a few seconds? wait for the soonest recovery, retry once
            now = time.time()
            waits = [t - now for k, t in _cooldown.items() if t > now and k.split(":", 1)[0] in names]
            if waits and min(waits) <= 20:
                time.sleep(min(waits) + 0.5)
                continue
        break
    LAST_PROVIDER = "none (showing document passages only)"
    raise LLMUnavailable("All LLM models failed or are cooling down: " + "; ".join(errors or ["all on cooldown"]))
