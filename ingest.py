"""Incrementally index data/raw (.pdf/.txt). Only documents whose content hash changed are re-embedded."""
import hashlib
import json
from pathlib import Path

import fitz  # pymupdf
import numpy as np

import embedder

RAW, INDEX = Path("data/raw"), Path("data/index")
DOCS, MANIFEST = INDEX / "docs", Path("data/manifest.json")
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
        if cached.get("sha256") == sha and cached.get("embedder") == embedder.ID:
            chunks, emb = cached["chunks"], np.load(cn)
        else:
            chunks = [{"page": p, "text": c} for p, t in read_pages(f) for c in chunk_words(t)]
            if not chunks:
                print(f"no text in {f.name} (scanned PDF?), skipping")
                continue
            emb = embedder.embed([c["text"] for c in chunks])
            cj.write_text(json.dumps({"sha256": sha, "embedder": embedder.ID, "chunks": chunks}))
            np.save(cn, emb)
            print(f"indexed {f.name}: {len(chunks)} chunks")
        for c in chunks:  # freshness metadata is attached at load time so it is never stale
            all_chunks.append({**c, "source": f.name, "label": meta.get("label", f.stem),
                               "url": meta.get("url", ""), "checked": meta.get("checked_at", "")[:10]})
        all_emb.append(emb)
    if not all_chunks:
        raise SystemExit("No documents indexed. Run `python fetch.py` or put PDFs in data/raw/.")
    np.save(INDEX / "emb.npy", np.vstack(all_emb))
    (INDEX / "chunks.json").write_text(json.dumps(all_chunks))
    print(f"index ready: {len(all_chunks)} chunks")


if __name__ == "__main__":
    main()
