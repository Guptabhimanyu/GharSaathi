"""Incrementally index data/raw (.pdf/.txt). Only documents whose content hash changed are re-embedded."""
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path

import fitz  # pymupdf
import numpy as np

import embedder

RAW, INDEX = Path("data/raw"), Path("data/index")
DOCS, MANIFEST = INDEX / "docs", Path("data/manifest.json")
CLEAN_VERSION = 2  # bump when text cleaning changes so cached chunks get rebuilt
# Some PDFs encode the "ti" ligature as a stray glyph (e.g. "applicaƟon", "meeƟng"); put "ti" back.
_LIG = re.compile("(?<=[A-Za-z])[\u019f\u0398\u03b8]|[\u019f\u0398\u03b8](?=[A-Za-z])")


def clean(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)  # also expands fi/fl/ff ligatures

    def rep(m):
        s = m.string
        around = s[m.start() - 1 : m.start()] + s[m.start() + 1 : m.start() + 2]
        return "TI" if around.isupper() else "ti"

    return _LIG.sub(rep, text)
def chunk_words(text, size=350, overlap=60):
    words = text.split()
    for i in range(0, len(words), size - overlap):
        piece = words[i : i + size]
        if len(piece) > 20:
            yield " ".join(piece)


def read_pages(path: Path):
    if path.suffix == ".pdf":
        for pno, page in enumerate(fitz.open(path), 1):
            yield pno, page.get_text()
    else:
        yield 1, path.read_text(errors="ignore")


def main():
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    DOCS.mkdir(parents=True, exist_ok=True)
    all_chunks, all_emb = [], []
    for f in sorted([*RAW.glob("*.pdf"), *RAW.glob("*.txt")]):
        meta = manifest.get(f.stem, {})
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        cj, cn = DOCS / f"{f.stem}.json", DOCS / f"{f.stem}.npy"
        cached = json.loads(cj.read_text()) if cj.exists() and cn.exists() else {}
        if cached.get("sha256") == sha and cached.get("embedder") == embedder.ID and cached.get("clean") == CLEAN_VERSION:
            chunks, emb = cached["chunks"], np.load(cn)
        else:
            chunks = [{"page": p, "text": c} for p, t in read_pages(f) for c in chunk_words(clean(t))]
            if not chunks:
                print(f"no text in {f.name} (scanned PDF?), skipping")
                continue
            emb = embedder.embed([c["text"] for c in chunks])
            cj.write_text(json.dumps({"sha256": sha, "embedder": embedder.ID, "clean": CLEAN_VERSION, "chunks": chunks}))
            np.save(cn, emb)
            print(f"indexed {f.name}: {len(chunks)} chunks")
        for c in chunks:  # freshness metadata is attached at load time so it is never stale
            all_chunks.append({**c, "source": f.name, "label": meta.get("label", f.stem),
                               "url": meta.get("url", ""), "checked": meta.get("checked_at", "")[:10]})
        all_emb.append(emb)
    if not all_chunks:
        raise SystemExit("No documents indexed. Run `python fetch.py` or put PDFs in data/raw/.")
    np.save(INDEX / "emb.tmp.npy", np.vstack(all_emb))  # write to temp files, then swap in atomically
    (INDEX / "chunks.tmp.json").write_text(json.dumps(all_chunks))
    os.replace(INDEX / "emb.tmp.npy", INDEX / "emb.npy")
    os.replace(INDEX / "chunks.tmp.json", INDEX / "chunks.json")
    print(f"index ready: {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()
