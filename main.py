#!/usr/bin/env python3
"""
Calibre bulk metadata updater + format embedder (idempotent).

What it does:
- Finds all books that have one of the target formats in the library.
- Filters to "English" OR "language missing" (so you can fix missing language too).
- For each book:
  - If already "good enough" and already processed (tracked), skip.
  - If "good enough" but not yet embedded, only embed metadata into EPUB.
  - Otherwise:
      - fetch-ebook-metadata -> OPF (+cover if found)
      - calibredb set_metadata (apply OPF to Calibre DB)
      - (optional) set cover field if downloaded
      - calibredb embed_metadata --only-formats <formats> (write metadata into files)
- If any step fails for a given book: record failure + continue.

State:
- Stored at: /drive/calibre/en_nonfiction/.calibre_metadata_state.json
- Idempotence:
  - We compute a stable hash of the *current DB metadata snapshot* we care about.
  - By default, a book is processed only once (per book id), regardless of later metadata changes.
  - If you want to allow reprocessing when metadata changes, set
    REPROCESS_ON_METADATA_CHANGE = True.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
import argparse
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple


# -----------------------
# User config
# -----------------------

DEFAULT_LIB = "/drive/calibre/en_nonfiction"
LIB = DEFAULT_LIB
STATE_PATH = os.path.join(LIB, ".calibre_metadata_state.json")

# File formats to operate on (calibre formats, lowercase)
DEFAULT_FORMATS = {"epub"}

# Treat as English if languages contains any of these (calibre often uses ISO639-3 like "eng")
ENGLISH_CODES = {"en", "eng", "en-us", "en-gb"}

# If a book has no language set at all, we still include it (so you can fix missing language)
INCLUDE_MISSING_LANGUAGE = True

# Metadata "good enough" scoring (tweak if you want)
MIN_SCORE_TO_SKIP_FETCH = 6

# Avoid hammering metadata sources (0.0 disables delay)
DELAY_BETWEEN_FETCHES_SECONDS = 0.35

# If True, a book is reprocessed when its metadata snapshot hash changes.
# If False, a book is processed only once (per book id).
REPROCESS_ON_METADATA_CHANGE = False


# -----------------------
# Helpers
# -----------------------


def log(msg: str) -> None:
    # consistent, simple logging
    print(msg, file=sys.stderr, flush=True)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required tool on PATH: {name}")


def run(
    cmd: List[str],
    *,
    check: bool = False,
    capture: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    log(f"[cmd] {' '.join(cmd)}")
    return subprocess.run(
        cmd,
        text=True,
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
        env=env,
    )


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="replace"))
    return h.hexdigest()


def stable_json_dumps(obj: Any) -> str:
    # Deterministic JSON for hashing
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def normalize_languages(val: Any) -> List[str]:
    # calibredb --for-machine often returns list; but be defensive
    if val is None:
        return []
    if isinstance(val, list):
        out = []
        for x in val:
            if x is None:
                continue
            out.append(str(x).strip().lower())
        return [x for x in out if x]
    s = str(val).strip().lower()
    return [s] if s else []


def is_english_or_missing(langs: List[str]) -> bool:
    if not langs:
        return INCLUDE_MISSING_LANGUAGE
    # Some installs might store "English" or "en_US" — normalize a bit
    for x in langs:
        x2 = x.replace("_", "-").lower()
        if x2 in ENGLISH_CODES:
            return True
        if x2.startswith("en-"):
            return True
        if x2 == "english":
            return True
    return False


def normalize_formats(val: Any) -> List[str]:
    if val is None:
        return []
    if isinstance(val, list):
        out = []
        for x in val:
            if x is None:
                continue
            s = str(x).strip().lower()
            if s:
                out.append(s)
        return out
    s = str(val).lower()
    return [x.strip() for x in s.replace(";", ",").split(",") if x.strip()]


def has_any_format(formats_val: Any, targets: set[str]) -> bool:
    fmts = normalize_formats(formats_val)
    if not fmts:
        return False
    return any(f in targets for f in fmts)


def normalize_identifiers(val: Any) -> Dict[str, str]:
    # Usually dict like {"isbn":"...", "asin":"..."}; convert to simple dict[str,str]
    out: Dict[str, str] = {}
    if isinstance(val, dict):
        for k, v in val.items():
            if v is None:
                continue
            ks = str(k).strip().lower()
            vs = str(v).strip()
            if ks and vs:
                out[ks] = vs
    return out


def metadata_snapshot(book: Dict[str, Any]) -> Dict[str, Any]:
    """
    Snapshot the DB metadata fields we care about for:
    - determining "good enough"
    - idempotence (hash)
    """
    identifiers = normalize_identifiers(book.get("identifiers"))
    langs = normalize_languages(book.get("languages"))

    # Normalize authors (may be list or string)
    authors_val = book.get("authors")
    if isinstance(authors_val, list):
        authors = [
            str(a).strip() for a in authors_val if a is not None and str(a).strip()
        ]
    else:
        a = str(authors_val or "").strip()
        authors = [a] if a else []

    # tags can also be list or string
    tags_val = book.get("tags")
    if isinstance(tags_val, list):
        tags = [str(t).strip() for t in tags_val if t is not None and str(t).strip()]
    else:
        s = str(tags_val or "").strip()
        tags = [x.strip() for x in s.split(",") if x.strip()] if s else []

    snap = {
        "title": str(book.get("title") or "").strip(),
        "authors": authors,
        "publisher": str(book.get("publisher") or "").strip(),
        "pubdate": str(
            book.get("pubdate") or ""
        ).strip(),  # calibre may store ISO-ish datetime
        "languages": langs,
        "isbn": str(book.get("isbn") or "").strip(),
        "identifiers": identifiers,  # already normalized
        "tags": tags,
        "comments_present": bool(str(book.get("comments") or "").strip()),
        "cover_present": bool(book.get("cover")),  # often True/False-ish
    }
    return snap


def snapshot_hash(snap: Dict[str, Any]) -> str:
    return sha256_text(stable_json_dumps(snap))


def score_good_enough(snap: Dict[str, Any]) -> Tuple[int, List[str]]:
    """
    Simple heuristic scoring to decide whether to skip online fetching.
    You can tweak MIN_SCORE_TO_SKIP_FETCH if you want stricter/looser.
    """
    score = 0
    reasons: List[str] = []

    # Required-ish
    if snap["title"]:
        score += 1
    else:
        reasons.append("missing title")

    if snap["authors"]:
        score += 1
    else:
        reasons.append("missing authors")

    # Valuable fields
    if snap["publisher"]:
        score += 1
    else:
        reasons.append("missing publisher")

    if snap["pubdate"]:
        score += 1
    else:
        reasons.append("missing pubdate")

    # Identifiers: give extra weight (helps matching)
    if snap["isbn"]:
        score += 2
    elif snap["identifiers"]:
        score += 2
    else:
        reasons.append("missing identifiers/isbn")

    if snap["tags"]:
        score += 1
    else:
        reasons.append("missing tags")

    if snap["comments_present"]:
        score += 1
    else:
        reasons.append("missing description/comments")

    if snap["cover_present"]:
        score += 1
    else:
        reasons.append("missing cover")

    return score, reasons


# -----------------------
# State handling
# -----------------------


@dataclasses.dataclass
class BookState:
    status: str  # "done" | "embedded_only" | "skipped_good_enough" | "failed"
    last_hash: str
    last_attempt_utc: str
    last_ok_utc: Optional[str] = None
    message: Optional[str] = None
    fail_count: int = 0


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"version": 1, "updated_at_utc": None, "books": {}}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, dict):
            raise ValueError("state file not a dict")
        obj.setdefault("version", 1)
        obj.setdefault("books", {})
        if not isinstance(obj["books"], dict):
            obj["books"] = {}
        return obj
    except Exception as e:
        raise SystemExit(f"Failed to read state file {STATE_PATH}: {e}")


def save_state(state: Dict[str, Any]) -> None:
    state["updated_at_utc"] = now_iso()
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, STATE_PATH)


def get_book_state(state: Dict[str, Any], book_id: int) -> Optional[BookState]:
    b = state.get("books", {}).get(str(book_id))
    if not isinstance(b, dict):
        return None
    try:
        return BookState(
            status=str(b.get("status") or ""),
            last_hash=str(b.get("last_hash") or ""),
            last_attempt_utc=str(b.get("last_attempt_utc") or ""),
            last_ok_utc=b.get("last_ok_utc"),
            message=b.get("message"),
            fail_count=int(b.get("fail_count") or 0),
        )
    except Exception:
        return None


def put_book_state(state: Dict[str, Any], book_id: int, bs: BookState) -> None:
    state.setdefault("books", {})
    state["books"][str(book_id)] = dataclasses.asdict(bs)


# -----------------------
# Calibre querying
# -----------------------


def list_candidate_books(target_formats: set[str]) -> List[Dict[str, Any]]:
    # Pull enough fields to:
    # - decide english/missing language
    # - compute snapshot hash + good-enough score
    fields = ",".join(
        [
            "id",
            "title",
            "authors",
            "publisher",
            "pubdate",
            "languages",
            "formats",
            "isbn",
            "identifiers",
            "tags",
            "comments",
            "cover",
            "last_modified",
        ]
    )

    if not target_formats:
        raise SystemExit("No target formats provided.")
    search_expr = " or ".join(f"formats:{f}" for f in sorted(target_formats))
    cp = run(
        [
            "calibredb",
            "--with-library",
            LIB,
            "list",
            "--for-machine",
            "--fields",
            fields,
            "--search",
            search_expr,
        ],
        capture=True,
        check=True,
    )

    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError as e:
        log("[fatal] Failed to parse JSON from calibredb list --for-machine")
        log(cp.stdout)
        raise SystemExit(str(e))

    if not isinstance(data, list):
        raise SystemExit(f"Unexpected JSON shape from calibredb list: {type(data)}")

    out: List[Dict[str, Any]] = []
    for b in data:
        if not isinstance(b, dict):
            continue
        if not has_any_format(b.get("formats"), target_formats):
            continue
        langs = normalize_languages(b.get("languages"))
        if not is_english_or_missing(langs):
            continue
        out.append(b)

    return out


# -----------------------
# Update flow per book
# -----------------------


def fetch_metadata_to_opf_and_cover(
    book: Dict[str, Any],
    opf_path: str,
    cover_path: str,
) -> Tuple[bool, str]:
    """
    Returns (ok, message). On success, opf_path is created, cover_path may or may not exist.
    """
    title = str(book.get("title") or "").strip()
    authors_val = book.get("authors")
    if isinstance(authors_val, list):
        authors = ", ".join(
            str(a) for a in authors_val if a is not None and str(a).strip()
        )
    else:
        authors = str(authors_val or "").strip()

    isbn = str(book.get("isbn") or "").strip()
    identifiers = normalize_identifiers(book.get("identifiers"))

    cmd = ["fetch-ebook-metadata", "--opf", opf_path, "--cover", cover_path]

    if isbn:
        cmd += ["--isbn", isbn]
    else:
        # Prefer strong identifiers if we have them
        for k, v in identifiers.items():
            cmd += ["--identifier", f"{k}:{v}"]
        # Fallback to title/authors
        if title:
            cmd += ["--title", title]
        if authors:
            cmd += ["--authors", authors]

    cp = run(cmd, capture=True, check=False)
    if cp.returncode != 0:
        msg = f"fetch-ebook-metadata failed rc={cp.returncode}"
        if cp.stderr:
            msg += f" stderr={cp.stderr.strip()[:500]}"
        return False, msg

    if not os.path.exists(opf_path) or os.path.getsize(opf_path) == 0:
        return False, "fetch-ebook-metadata produced no OPF"
    return True, "fetched"


def apply_opf_to_calibre_db(book_id: int, opf_path: str) -> Tuple[bool, str]:
    cp = run(
        ["calibredb", "--with-library", LIB, "set_metadata", str(book_id), opf_path],
        capture=True,
        check=False,
    )
    if cp.returncode != 0:
        msg = f"set_metadata failed rc={cp.returncode}"
        if cp.stderr:
            msg += f" stderr={cp.stderr.strip()[:500]}"
        return False, msg
    return True, "metadata applied"


def apply_cover_to_calibre_db(book_id: int, cover_path: str) -> Tuple[bool, str]:
    if not os.path.exists(cover_path) or os.path.getsize(cover_path) == 0:
        return True, "no cover downloaded"

    # Set cover via --field cover:<path>
    cp = run(
        [
            "calibredb",
            "--with-library",
            LIB,
            "set_metadata",
            str(book_id),
            "--field",
            f"cover:{cover_path}",
        ],
        capture=True,
        check=False,
    )
    if cp.returncode != 0:
        msg = f"cover set failed rc={cp.returncode}"
        if cp.stderr:
            msg += f" stderr={cp.stderr.strip()[:500]}"
        return False, msg

    return True, "cover applied"


def embed_metadata_into_formats(
    book_id: int, target_formats: set[str]
) -> Tuple[bool, str]:
    # Update metadata in-place for target formats only
    if not target_formats:
        return False, "no target formats"
    fmt_arg = ",".join(sorted(f.upper() for f in target_formats))
    cp = run(
        [
            "calibredb",
            "--with-library",
            LIB,
            "embed_metadata",
            "--only-formats",
            fmt_arg,
            str(book_id),
        ],
        capture=True,
        check=False,
    )
    if cp.returncode != 0:
        msg = f"embed_metadata failed rc={cp.returncode}"
        if cp.stderr:
            msg += f" stderr={cp.stderr.strip()[:500]}"
        return False, msg
    return True, "embedded"


def process_one_book(
    state: Dict[str, Any],
    book: Dict[str, Any],
    workdir: str,
    target_formats: set[str],
) -> None:
    book_id = int(book["id"])
    title = str(book.get("title") or "").strip()

    snap = metadata_snapshot(book)
    h = snapshot_hash(snap)

    prev = get_book_state(state, book_id)
    if prev and prev.status in {"done", "skipped_good_enough", "embedded_only"}:
        if (not REPROCESS_ON_METADATA_CHANGE) or (prev.last_hash == h):
            reason = (
                "already processed"
                if not REPROCESS_ON_METADATA_CHANGE
                else "already processed for current metadata hash"
            )
            log(f"[skip] id={book_id} title={title!r} ({reason})")
            return

    score, reasons = score_good_enough(snap)
    good_enough = (
        (score >= MIN_SCORE_TO_SKIP_FETCH)
        and bool(snap["title"])
        and bool(snap["authors"])
    )

    # If metadata is already good enough, we still want to ensure EPUB has embedded metadata at least once.
    # If we don't have a record for this hash, we do an embed-only pass.
    if good_enough:
        log(
            f"[ok?] id={book_id} title={title!r} score={score} -> good enough; embedding only"
        )
        ok_embed, msg_embed = embed_metadata_into_formats(book_id, target_formats)

        bs = BookState(
            status="embedded_only" if ok_embed else "failed",
            last_hash=h,
            last_attempt_utc=now_iso(),
            last_ok_utc=now_iso() if ok_embed else (prev.last_ok_utc if prev else None),
            message=("good enough; embedded" if ok_embed else msg_embed)
            + ("" if ok_embed else f" (good enough reasons: {', '.join(reasons)})"),
            fail_count=(0 if ok_embed else ((prev.fail_count if prev else 0) + 1)),
        )
        put_book_state(state, book_id, bs)
        save_state(state)

        if ok_embed:
            log(f"[done] id={book_id} title={title!r} (good enough; embedded)")
        else:
            log(f"[fail] id={book_id} title={title!r} ({msg_embed})")
        return

    # Otherwise, attempt full fetch -> apply -> embed
    log(
        f"[work] id={book_id} title={title!r} score={score} (not good enough; will fetch). missing: {', '.join(reasons)}"
    )

    opf_path = os.path.join(workdir, f"{book_id}.opf")
    cover_path = os.path.join(workdir, f"{book_id}.cover.jpg")

    ok_fetch, msg_fetch = fetch_metadata_to_opf_and_cover(book, opf_path, cover_path)
    if not ok_fetch:
        bs = BookState(
            status="failed",
            last_hash=h,
            last_attempt_utc=now_iso(),
            last_ok_utc=prev.last_ok_utc if prev else None,
            message=msg_fetch,
            fail_count=(prev.fail_count if prev else 0) + 1,
        )
        put_book_state(state, book_id, bs)
        save_state(state)
        log(f"[skip] id={book_id} title={title!r} ({msg_fetch})")
        return

    if DELAY_BETWEEN_FETCHES_SECONDS > 0:
        time.sleep(DELAY_BETWEEN_FETCHES_SECONDS)

    ok_set, msg_set = apply_opf_to_calibre_db(book_id, opf_path)
    if not ok_set:
        bs = BookState(
            status="failed",
            last_hash=h,
            last_attempt_utc=now_iso(),
            last_ok_utc=prev.last_ok_utc if prev else None,
            message=msg_set,
            fail_count=(prev.fail_count if prev else 0) + 1,
        )
        put_book_state(state, book_id, bs)
        save_state(state)
        log(f"[skip] id={book_id} title={title!r} ({msg_set})")
        return

    # Best-effort cover application
    ok_cov, msg_cov = apply_cover_to_calibre_db(book_id, cover_path)
    if not ok_cov:
        log(f"[warn] id={book_id} title={title!r} ({msg_cov})")

    # Now embed metadata into the EPUB file(s)
    ok_embed, msg_embed = embed_metadata_into_formats(book_id, target_formats)
    if not ok_embed:
        bs = BookState(
            status="failed",
            last_hash=h,
            last_attempt_utc=now_iso(),
            last_ok_utc=prev.last_ok_utc if prev else None,
            message=msg_embed,
            fail_count=(prev.fail_count if prev else 0) + 1,
        )
        put_book_state(state, book_id, bs)
        save_state(state)
        log(f"[skip] id={book_id} title={title!r} ({msg_embed})")
        return

    # Re-read the book list entry to capture the post-update DB snapshot hash,
    # so future runs are idempotent even after metadata changed.
    # We do a lightweight "list --search id:X" call.
    refreshed = refresh_one_book(book_id)
    new_snap = metadata_snapshot(refreshed) if refreshed else snap
    new_hash = snapshot_hash(new_snap)

    bs = BookState(
        status="done",
        last_hash=new_hash,
        last_attempt_utc=now_iso(),
        last_ok_utc=now_iso(),
        message="fetched+applied+embedded",
        fail_count=0,
    )
    put_book_state(state, book_id, bs)
    save_state(state)
    log(f"[done] id={book_id} title={title!r} (updated + embedded)")


def refresh_one_book(book_id: int) -> Optional[Dict[str, Any]]:
    fields = ",".join(
        [
            "id",
            "title",
            "authors",
            "publisher",
            "pubdate",
            "languages",
            "formats",
            "isbn",
            "identifiers",
            "tags",
            "comments",
            "cover",
            "last_modified",
        ]
    )
    cp = run(
        [
            "calibredb",
            "--with-library",
            LIB,
            "list",
            "--for-machine",
            "--fields",
            fields,
            "--search",
            f"id:{book_id}",
        ],
        capture=True,
        check=False,
    )
    if cp.returncode != 0 or not cp.stdout:
        return None
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


# -----------------------
# Main
# -----------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibre bulk metadata updater + format embedder"
    )
    parser.add_argument(
        "--library",
        dest="library",
        default=DEFAULT_LIB,
        help="Path to Calibre library (default: %(default)s)",
    )
    parser.add_argument(
        "--formats",
        dest="formats",
        default=",".join(sorted(DEFAULT_FORMATS)),
        help="Comma-separated Calibre formats to process (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> int:
    require_tool("calibredb")
    require_tool("fetch-ebook-metadata")

    args = parse_args()
    global LIB, STATE_PATH
    LIB = args.library
    STATE_PATH = os.path.join(LIB, ".calibre_metadata_state.json")
    target_formats = {
        f.strip().lower() for f in args.formats.split(",") if f.strip()
    }
    if not target_formats:
        raise SystemExit("No formats specified. Use --formats epub,pdf")

    if not os.path.isdir(LIB):
        raise SystemExit(f"Library path does not exist or is not a directory: {LIB}")

    state = load_state()
    books = list_candidate_books(target_formats)

    log(f"[info] library={LIB}")
    log(f"[info] state={STATE_PATH}")
    log(
        f"[info] candidates={len(books)} (formats={','.join(sorted(target_formats))} + English-or-missing-language)"
    )

    ok = 0
    fail = 0
    skipped = 0

    with tempfile.TemporaryDirectory(prefix="calibre-meta-", dir=None) as workdir:
        for b in books:
            book_id = int(b["id"])
            title = str(b.get("title") or "").strip()
            try:
                prev = get_book_state(state, book_id)
                before_hash = snapshot_hash(metadata_snapshot(b))
                if prev and prev.status in {"done", "skipped_good_enough", "embedded_only"}:
                    if (not REPROCESS_ON_METADATA_CHANGE) or (prev.last_hash == before_hash):
                        skipped += 1
                        reason = (
                            "already processed"
                            if not REPROCESS_ON_METADATA_CHANGE
                            else "already processed for current metadata hash"
                        )
                        log(f"[skip] id={book_id} title={title!r} ({reason})")
                        continue

                process_one_book(state, b, workdir, target_formats)

                # classify outcome based on fresh state
                after = get_book_state(state, book_id)
                if after and after.status == "done":
                    ok += 1
                elif after and after.status == "failed":
                    fail += 1
                else:
                    # should not happen often, but keep counters sane
                    skipped += 1

            except Exception as e:
                fail += 1
                prev = get_book_state(state, book_id)
                snap = metadata_snapshot(b)
                h = snapshot_hash(snap)
                bs = BookState(
                    status="failed",
                    last_hash=h,
                    last_attempt_utc=now_iso(),
                    last_ok_utc=prev.last_ok_utc if prev else None,
                    message=f"exception: {e}",
                    fail_count=(prev.fail_count if prev else 0) + 1,
                )
                put_book_state(state, book_id, bs)
                save_state(state)
                log(f"[fail] id={book_id} title={title!r} (exception: {e})")

    log(f"[summary] done_ok={ok} done_failed={fail} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
