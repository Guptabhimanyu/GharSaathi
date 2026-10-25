"""One call to fetch the latest official documents and update the index. Safe to run repeatedly."""
import fetch
import ingest


def refresh(force: bool = False) -> list[str]:
    changed = fetch.fetch_all(force=force)
    ingest.main()
    return changed


if __name__ == "__main__":
    print("changed:", refresh() or "nothing")
