"""Speech-in / speech-out helpers for Hindi + English. Everything degrades gracefully if a piece is missing."""
import asyncio
import io
import os
import re

from openai import OpenAI

import llm

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_ARABIC = re.compile(r"[\u0600-\u06FF]")
_local = None


def detect_lang(text: str) -> str:
    """'hi' if the text is mostly Devanagari, else 'en'. Decides the language of the answer."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "en"
    return "hi" if sum(bool(_DEVANAGARI.match(c)) for c in letters) / len(letters) > 0.3 else "en"


def transcribe(audio: bytes, lang_hint: str | None = None) -> str:
    """Groq Whisper first (free tier); local faster-whisper if installed and Groq fails."""
    err = None
    if os.getenv("GROQ_API_KEY"):
        try:
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=os.environ["GROQ_API_KEY"],
                            timeout=60, max_retries=0)

            def run(language):
                kw = dict(model=os.getenv("GROQ_STT_MODEL", "whisper-large-v3"), file=("speech.wav", audio),
                          response_format="text", temperature=0)
                if language:
                    kw["language"] = language
                return str(client.audio.transcriptions.create(**kw)).strip()

            text = run(lang_hint)
            if not lang_hint and _ARABIC.search(text):  # Whisper sometimes writes Hindi speech in Urdu script
                text = run("hi")
            if text:
                return text
        except Exception as e:  # rate limit, network, etc.
            err = e
    try:
        global _local
        from faster_whisper import WhisperModel
        _local = _local or WhisperModel("small", compute_type="int8")
        segments, _ = _local.transcribe(io.BytesIO(audio), language=lang_hint)
        return " ".join(s.text for s in segments).strip()
    except ImportError:
        raise RuntimeError(f"Speech recognition unavailable (Groq failed: {err}; faster-whisper not installed).")


def to_english(text: str) -> str:
    """Translate a Hindi question to English so it can search the English documents."""
    return llm.complete(
        "Translate the user's Hindi text to English. Output only the translation. Keep all numbers and amounts exact.",
        [{"role": "user", "content": text}], max_tokens=300)


def speak(text: str, lang: str) -> bytes | None:
    """Optional text-to-speech via edge-tts (unofficial, may break). Returns mp3 bytes or None."""
    try:
        import edge_tts
    except ImportError:
        return None
    voice = {"hi": "hi-IN-SwaraNeural", "en": "en-IN-NeerjaNeural"}[lang]
    clean = re.sub(r"\[\d+\]|[*#_`>|]", "", text)[:1200]

    async def run():
        out = bytearray()
        async for ch in edge_tts.Communicate(clean, voice).stream():
            if ch["type"] == "audio":
                out += ch["data"]
        return bytes(out)

    try:
        return asyncio.run(run()) or None
    except Exception:
        return None
