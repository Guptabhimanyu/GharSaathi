"""Fetch the latest official documents and update the index. Safe to run repeatedly."""
import threading
import time

import fetch
import ingest

_lock = threading.Lock()
_state = {"running": False, "last": 0.0, "changed": False}


def refresh(force: bool = False) -> list[str]:
    changed = fetch.fetch_all(force=force)
    ingest.main()
    return changed


def _bg():
    try:
        refresh()
        _state["changed"] = True  # index/manifest were rewritten: tell the app to reload them
    except (Exception, SystemExit) as e:
        print("background refresh failed, keeping existing index:", e)
    finally:
        _state["last"] = time.time()
        _state["running"] = False


def start_background(interval: float = 24 * 3600, now: bool = False) -> bool:
    """Start a refresh in a thread at most once per `interval` seconds. Never blocks the page."""
    with _lock:
        if _state["running"] or (not now and time.time() - _state["last"] < interval):
            return False
        _state["running"] = True
    threading.Thread(target=_bg, daemon=True).start()
    return True


def is_running() -> bool:
    return _state["running"]


def consume_changed() -> bool:
    changed, _state["changed"] = _state["changed"], False
    return changed


if __name__ == "__main__":
    print("changed:", refresh() or "nothing")
