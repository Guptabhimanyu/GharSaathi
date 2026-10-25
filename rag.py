"""Retrieval + grounded generation."""
import json
from pathlib import Path

import numpy as np

import embedder
import llm
from user_profile import Profile

INDEX = Path("data/index")

SYSTEM = """You help people in India understand government housing schemes, using ONLY the numbered context passages provided.
Rules:
- Base every claim on the context and cite it like [1] or [2][3].
- If the context does not answer something, say "the documents I have don't cover this" rather than guessing.
- Never invent eligibility rules, amounts, or deadlines.
- If passages conflict, prefer the more recent / PMAY-U 2.0 document and say that older documents differ.
- Use plain, simple language. Mention that rules can change and the person should confirm at the official portal or their urban local body.
- Be brief: lead with the direct answer, then only what this person needs. No tables, no repeated caveats.
- Copy numbers and amounts exactly from the passages. Anything not in the passages must be labelled "general tip (not from the documents)" or left out.
- You are not official advice."""


class Retriever:
    def __init__(self):
        self.emb = np.load(INDEX / "emb.npy")
        self.chunks = json.loads((INDEX / "chunks.json").read_text())

    def search(self, query: str, k: int = 5):
        q = embedder.embed([query])[0]
        top = np.argsort(-(self.emb @ q))[:k]
        return [self.chunks[i] for i in top]

    def search_many(self, queries, k_each=3, limit=8):
        seen, out = set(), []
        for q in queries:
            for c in self.search(q, k_each):
                key = (c["source"], c["page"], c["text"][:60])
                if key not in seen:
                    seen.add(key)
                    out.append(c)
        return out[:limit]


def _context(chunks) -> str:
    return "\n\n".join(f"[{i}] ({c.get('label', c['source'])}, p.{c['page']}, checked {c.get('checked') or 'n/a'})\n{c['text']}" for i, c in enumerate(chunks, 1))


def _llm(messages, max_tokens=1200) -> str:
    return llm.complete(SYSTEM, messages, max_tokens)


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
    try:
        return _llm(messages, max_tokens)
    except llm.LLMUnavailable as e:
        return _extractive(chunks, reason=str(e))


def personalized_plan(retriever: Retriever, profile: Profile):
    chunks = retriever.search_many(profile.queries())
    prompt = (
        f"Person's profile:\n{profile.summary()}\n\nContext:\n{_context(chunks)}\n\n"
        "Write a SHORT guide, under 180 words, with exactly these three headings:\n"
        "**Your best options** (max 3 bullets, one line each)\n"
        "**Check before applying** (max 3 bullets, only conditions that depend on facts you don't know)\n"
        "**Next steps** (max 4 short numbered steps)\n"
        "Mention only what is relevant to this profile. Do not repeat the profile back. "
        "If something important is not in the documents, add one final line starting 'Not covered by the documents:'."
    )
    return _answer([{"role": "user", "content": prompt}], chunks, 700), chunks


def chat_reply(retriever: Retriever, profile: Profile, history: list[dict], question: str,
               lang: str = "en", search_query: str | None = None):
    sq = search_query or question  # English text used for searching the (English) documents
    chunks = retriever.search_many([sq, f"{sq} {profile.occupation} {profile.need}"], k_each=4, limit=6)
    answer_in = "Hindi (Devanagari script)" if lang == "hi" else "English"
    prompt = (
        f"Person's profile:\n{profile.summary()}\n\nContext:\n{_context(chunks)}\n\nQuestion: {question}\n\n"
        f"Answer in {answer_in}. Be brief: 2-4 sentences or at most 4 short bullets, direct answer first. "
        f"Give more detail only if the person asks for it. Keep numbers, amounts and citation markers like [1] unchanged."
    )
    return _answer(history + [{"role": "user", "content": prompt}], chunks, 800), chunks


def loan_options(retriever: Retriever, profile: Profile):
    chunks = retriever.search_many(profile.loan_queries(), k_each=3, limit=9)
    prompt = (
        f"Person's profile:\n{profile.summary()}\n\nContext:\n{_context(chunks)}\n\n"
        "Suggest at most 3 loan ROUTES worth exploring for this person, best fit first, under 200 words in total. "
        "A route is a type of loan or loan-linked scheme from the context (e.g. a subsidised home loan, a guarantee-backed "
        "loan), NOT a specific bank's product. For each route write three short lines: what it is; why it may fit; what to check.\n"
        "- Use only the context for scheme numbers, caps and eligibility, and cite them.\n"
        "- Never state a bank's interest rate or call a lender 'the best'.\n"
        "- Consider their income proof and existing EMIs. If a loan looks hard for them, say so and point to non-loan routes in the documents.\n"
        "- End with exactly one line: 'General information, not financial advice. Compare written quotes from 2-3 regulated lenders, "
        "and never pay anyone to guarantee a subsidy.'"
    )
    return _answer([{"role": "user", "content": prompt}], chunks, 700), chunks
