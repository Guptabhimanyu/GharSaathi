"""Download/refresh scheme documents listed in sources.json (official .gov.in / .nic.in domains only).

Uses ETag / Last-Modified so unchanged files cost almost nothing, and a content hash so we only
re-embed documents whose text actually changed. If a site is down we keep the last good copy.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

RAW, MANIFEST = Path("data/raw"), Path("data/manifest.json")
ALLOWED_SUFFIXES = (".gov.in", ".nic.in")
HEADERS = {"User-Agent": "Mozilla/5.0 (housing-guide research project)"}


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        t.decompose()
    lines = (ln.strip() for ln in soup.get_text("\n").splitlines())
    return "\n".join(ln for ln in lines if ln)


def fetch_all(force: bool = False) -> list[str]:
    RAW.mkdir(parents=True, exist_ok=True)
    manifest, changed = load_manifest(), []
    for s in json.loads(Path("sources.json").read_text()):
        host = urlparse(s["url"]).hostname or ""
        if not host.endswith(ALLOWED_SUFFIXES):
            print(f"skip {s['id']}: {host} is not an allowed official domain")
            continue
        m = manifest.get(s["id"], {})
        headers = dict(HEADERS)
        if not force and m.get("etag"):
            headers["If-None-Match"] = m["etag"]
        if not force and m.get("last_modified"):
            headers["If-Modified-Since"] = m["last_modified"]
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        try:
            r = requests.get(s["url"], headers=headers, timeout=60)
        except requests.RequestException as e:
            print(f"warn {s['id']}: {type(e).__name__}; keeping cached copy")
            continue
        if r.status_code == 304:
            m["checked_at"] = now
            manifest[s["id"]] = m
            continue
        if r.status_code != 200:
            print(f"warn {s['id']}: HTTP {r.status_code}; keeping cached copy")
            continue
        if s["type"] == "html":
            data, ext = html_to_text(r.text).encode(), "txt"
        else:
            data, ext = r.content, "pdf"
            if not data.startswith(b"%PDF"):
                print(f"warn {s['id']}: response is not a PDF; keeping cached copy")
                continue
        digest = hashlib.sha256(data).hexdigest()
        m.update(label=s["label"], url=s["url"], etag=r.headers.get("ETag"),
                 last_modified=r.headers.get("Last-Modified"), checked_at=now)
        if digest != m.get("sha256"):
            (RAW / f"{s['id']}.{ext}").write_bytes(data)
            m.update(sha256=digest, changed_at=now)
            changed.append(s["id"])
            print(f"updated {s['id']}")
        manifest[s["id"]] = m
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    return changed


if __name__ == "__main__":
    print("changed:", fetch_all(force=False) or "nothing")
