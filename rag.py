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


def _extractive(chunks, n=3) -> str:
    lines = ["⚠️ The AI assistant is temporarily unavailable (free-tier limits reached). Here are the most relevant "
             "passages from the official documents instead. यहाँ आधिकारिक दस्तावेज़ों के सबसे प्रासंगिक अंश दिए गए हैं।\n"]
    for c in chunks[:n]:
        snippet = c["text"].replace("\n", " ")[:600].rsplit(" ", 1)[0] + "…"
        lines.append(f"**{c.get('label', c['source'])}, p.{c['page']}**\n> {snippet}\n")
    lines.append("Please try again in a minute.")
    return "\n".join(lines)


def _answer(messages, chunks, max_tokens=1200) -> str:
    try:
        return _llm(messages, max_tokens)
    except llm.LLMUnavailable:
        return _extractive(chunks)


def personalized_plan(retriever: Retriever, profile: Profile):
    chunks = retriever.search_many(profile.queries())
    prompt = (
        f"Person's profile:\n{profile.summary()}\n\nContext:\n{_context(chunks)}\n\n"
        "Write a short personalized guide with these headings:\n"
        "1. What you may be able to look into (options that fit this profile)\n"
        "2. What you must still check (eligibility conditions that depend on facts I don't know)\n"
        "3. Documents and next steps\n"
        "4. What the documents don't tell me"
    )
    return _answer([{"role": "user", "content": prompt}], chunks, 1500), chunks


def chat_reply(retriever: Retriever, profile: Profile, history: list[dict], question: str,
               lang: str = "en", search_query: str | None = None):
    sq = search_query or question  # English text used for searching the (English) documents
    chunks = retriever.search_many([sq, f"{sq} {profile.occupation} {profile.need}"], k_each=4, limit=6)
    answer_in = "Hindi (Devanagari script)" if lang == "hi" else "English"
    prompt = (
        f"Person's profile:\n{profile.summary()}\n\nContext:\n{_context(chunks)}\n\nQuestion: {question}\n\n"
        f"Answer in {answer_in}. Keep numbers, amounts and citation markers like [1] unchanged."
    )
    return _answer(history + [{"role": "user", "content": prompt}], chunks), chunks


def loan_options(retriever: Retriever, profile: Profile):
    chunks = retriever.search_many(profile.loan_queries(), k_each=3, limit=9)
    prompt = (
        f"Person's profile:\n{profile.summary()}\n\nContext:\n{_context(chunks)}\n\n"
        "Suggest the 2-3 loan ROUTES most worth exploring for this person, best fit first. A route is a type of loan or "
        "loan-linked scheme described in the context (for example a subsidised home loan, or a guarantee-backed loan), "
        "NOT a specific bank's product. For each route give: what it is; why it may fit this profile; "
        "what to verify before applying. Rules:\n"
        "- Use only the context for scheme numbers, caps and eligibility; cite them.\n"
        "- Do not state any bank's interest rate or call any lender 'the best'. Tell the person to get written quotes "
        "from at least 2-3 regulated lenders and compare total cost.\n"
        "- Take their income proof and existing EMIs into account. If a loan looks hard for them, say so honestly "
        "and mention the non-loan routes from the documents (e.g. rental housing).\n"
        "- Warn briefly about paying anyone to 'guarantee' a subsidy; applications go through the official portal and lenders.\n"
        "- Say this is general information, not financial advice."
    )
    return _answer([{"role": "user", "content": prompt}], chunks, 1500), chunks
