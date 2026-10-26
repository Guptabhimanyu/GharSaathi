"""Retrieval + grounded generation."""
import json
import os
import time
from pathlib import Path

import numpy as np

import embedder
import llm
from user_profile import Profile

INDEX = Path("data/index")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))  # below this similarity a passage is treated as unrelated
NOTICE_PREFIXES = ("⚠️", "ℹ️")  # system notices that must not be stored as chat history
OUT_OF_SCOPE = {
    "en": "ℹ️ I can only help with government housing schemes (rental housing, home loans and subsidies), using the official "
          "documents I have. Try for example: *What documents do I need for rental housing?*",
    "hi": "ℹ️ मैं केवल सरकारी आवास योजनाओं (किराये का मकान, होम लोन और सब्सिडी) के बारे में, अपने आधिकारिक दस्तावेज़ों के "
          "आधार पर मदद कर सकता हूँ। जैसे पूछिए: *किराये के मकान के लिए कौन से दस्तावेज़ चाहिए?*",
}

SYSTEM = """You help people in India understand government housing schemes, using ONLY the numbered context passages provided.
Rules:
- Base every claim on the context and cite it like [1] or [2][3].
- If the context does not answer something, say "the documents I have don't cover this" rather than guessing.
- Never invent eligibility rules, amounts, or deadlines.
- If passages conflict, prefer the more recent / PMAY-U 2.0 document and say that older documents differ.
- Use plain, simple language. Mention that rules can change and the person should confirm at the official portal or their urban local body.
- Before recommending any route, compare its income limit and other conditions in the passages with the person's annual income and situation. Never recommend a route they do not qualify for.
- Be brief: lead with the direct answer, then only what this person needs. No tables, no repeated caveats.
- Copy numbers and amounts exactly from the passages. Anything not in the passages must be labelled "general tip (not from the documents)" or left out.
- You are not official advice."""


class Retriever:
    def __init__(self):
        for _ in range(3):  # a background refresh may be swapping files in; retry until they agree
            self.emb = np.load(INDEX / "emb.npy")
            self.chunks = json.loads((INDEX / "chunks.json").read_text())
            if len(self.chunks) == len(self.emb):
                break
            time.sleep(1)

    def search(self, query: str, k: int = 5):
        q = embedder.embed([query])[0]
        scores = self.emb @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def search_many(self, queries, k_each=3, limit=8, diverse=False):
        """diverse=True takes the best hits of each query in turn (round robin), so no single topic crowds out the rest."""
        per_query = [[c for c in self.search(q, k_each) if c["score"] >= MIN_SCORE] for q in queries]
        seen, out = set(), []

        def add(c):
            key = (c["source"], c["page"], c["text"][:60])
            if key not in seen:
                seen.add(key)
                out.append(c)

        if diverse:
            for rank in range(k_each):
                for hits in per_query:
                    if rank < len(hits):
                        add(hits[rank])
            return out[:limit]
        for hits in per_query:
            for c in hits:
                add(c)
        out.sort(key=lambda c: -c["score"])
        return out[:limit]


def _context(chunks) -> str:
    return "\n\n".join(f"[{i}] ({c.get('label', c['source'])}, p.{c['page']}, checked {c.get('checked') or 'n/a'})\n{c['text']}" for i, c in enumerate(chunks, 1))


def _llm(messages, max_tokens=1200) -> str:
    return llm.complete(SYSTEM, messages, max_tokens)


def _lang_line(lang: str) -> str:
    return "Write everything in Hindi (Devanagari script), including headings and any closing line. " if lang == "hi" else ""


def _extractive(chunks, n=3, reason: str = "") -> str:
    lines = ["⚠️ The AI assistant is temporarily unavailable (free-tier limits reached). Here are the most relevant "
             "passages from the official documents instead. यहाँ आधिकारिक दस्तावेज़ों के सबसे प्रासंगिक अंश दिए गए हैं।\n"]
    for c in chunks[:n]:
        snippet = c["text"].replace("\n", " ")[:600].rsplit(" ", 1)[0] + "…"
        lines.append(f"**{c.get('label', c['source'])}, p.{c['page']}**\n> {snippet}\n")
    lines.append("Please try again in a minute.")
    if reason:
        lines.append(f"\n*Technical details: {reason[:500]}*")
    return "\n".join(lines)


def _answer(messages, chunks, max_tokens=1200) -> str:
    if not chunks:  # nothing relevant retrieved: don't call the model at all
        return OUT_OF_SCOPE["en"]
    try:
        return _llm(messages, max_tokens)
    except llm.LLMUnavailable as e:
        return _extractive(chunks, reason=str(e))


def personalized_plan(retriever: Retriever, profile: Profile, lang: str = "en"):
    chunks = retriever.search_many(profile.queries(), k_each=2, limit=7, diverse=True)
    prompt = (
        f"Person's profile:\n{profile.summary()}\n\nContext:\n{_context(chunks)}\n\n"
        f"{_lang_line(lang)}Write a SHORT guide, under 180 words, with exactly these three headings:\n"
        "Begin with one line: 'Income band: ...' stating which band the person's ANNUAL income falls in, using only the limits in the "
        "passages (if the passages give no limits, write 'Income band: not stated in the documents').\n"
        "**Your best options** (max 3 bullets, one line each)\n"
        "**Check before applying** (max 3 bullets, only conditions that depend on facts you don't know)\n"
        "**Next steps** (max 4 short numbered steps)\n"
        f"Stay on the person's stated need ({profile.need}). If it is a rental, do NOT list buying or building routes; instead end "
        "the 'Next steps' with one line saying they can ask about owning a house later. If it is to own a house, focus on ownership routes. "
        "If the city does not belong to the stated state, begin with one line saying so and asking them to correct it. "
        "Mention only what is relevant to this profile. Do not repeat the profile back. "
        "If something important is not in the documents, add one final line starting 'Not covered by the documents:'."
    )
    return _answer([{"role": "user", "content": prompt}], chunks, 1100 if lang == "hi" else 700), chunks


def chat_reply(retriever: Retriever, profile: Profile, history: list[dict], question: str,
               lang: str = "en", search_query: str | None = None):
    sq = search_query or question  # English text used for searching the (English) documents
    raw = retriever.search(sq, 3)  # relevance gate uses the question alone (profile words would inflate scores)
    if not any(c["score"] >= MIN_SCORE for c in raw):
        return OUT_OF_SCOPE.get(lang, OUT_OF_SCOPE["en"]), raw[:1]
    chunks = retriever.search_many([sq, f"{sq} {profile.occupation} {profile.need}"], k_each=4, limit=6)
    answer_in = "Hindi (Devanagari script)" if lang == "hi" else "English"
    prompt = (
        f"Person's profile:\n{profile.summary()}\n\nContext:\n{_context(chunks)}\n\nQuestion: {question}\n\n"
        f"Answer in {answer_in}. Be brief: 2-4 sentences or at most 4 short bullets, direct answer first. "
        f"Give more detail only if the person asks for it. Keep numbers, amounts and citation markers like [1] unchanged."
    )
    return _answer(history + [{"role": "user", "content": prompt}], chunks, 800), chunks


def loan_options(retriever: Retriever, profile: Profile, lang: str = "en"):
    chunks = retriever.search_many(profile.loan_queries(), k_each=2, limit=6, diverse=True)
    prompt = (
        f"Person's profile:\n{profile.summary()}\n\nContext:\n{_context(chunks)}\n\n"
        f"{_lang_line(lang)}Suggest at most 3 loan ROUTES worth exploring for this person, best fit first, under 200 words in total. "
        "A route is a type of loan or loan-linked scheme from the context (e.g. a subsidised home loan, a guarantee-backed "
        "loan), NOT a specific bank's product. For each route write three short lines: what it is; why it may fit; what to check.\n"
        "- Use only the context for scheme numbers, caps and eligibility, and cite them.\n"
        "- Never state a bank's interest rate or call a lender 'the best'.\n"
        "- Consider their income proof and existing EMIs. If a loan looks hard for them, say so and point to non-loan routes in the documents.\n"
        "- End with exactly one line: 'General information, not financial advice. Compare written quotes from 2-3 regulated lenders, "
        "and never pay anyone to guarantee a subsidy.'"
    )
    return _answer([{"role": "user", "content": prompt}], chunks, 1100 if lang == "hi" else 700), chunks
