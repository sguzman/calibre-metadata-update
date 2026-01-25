#!/usr/bin/env python3
# update_calibre_metadata.py
#
# For every English book with an EPUB format in a Calibre library:
# - Fetch updated metadata from the internet (and a cover if available)
# - Apply it to the Calibre database
# - If anything fails for a book, log it and continue

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any


LIB = "/drive/calibre/en_nonfiction"

# Calibre search query language: English + has EPUB
SEARCH_EXPR = "languages:eng and formats:epub"


def run(
    cmd: list[str], *, check: bool = False, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    # Extensive logging (you asked for it in general, and it's useful here)
    print(f"\n[cmd] {' '.join(cmd)}", file=sys.stderr)
    return subprocess.run(
        cmd,
        text=True,
        check=check,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required tool on PATH: {name}")


def get_books() -> list[dict[str, Any]]:
    # Use --for-machine JSON output so we don't have to parse human text.
    # Fields include isbn/identifiers so fetch-ebook-metadata can match better.
    cp = run(
        [
            "calibredb",
            "--with-library",
            LIB,
            "list",
            "--for-machine",
            "--fields",
            "id,title,authors,isbn,identifiers,languages,formats",
            "--search",
            SEARCH_EXPR,
        ],
        capture=True,
        check=True,
    )
    try:
        data = json.loads(cp.stdout)
    except json.JSONDecodeError as e:
        print(
            "[error] Failed to parse JSON from calibredb list --for-machine",
            file=sys.stderr,
        )
        print(cp.stdout, file=sys.stderr)
        raise SystemExit(str(e))

    if not isinstance(data, list):
        raise SystemExit(f"Unexpected JSON shape from calibredb list: {type(data)}")

    return data


def normalize_identifiers(identifiers: Any) -> list[str]:
    # calibredb "identifiers" usually comes back as a dict like {"isbn":"...", "asin":"..."}
    # We convert to fetch-ebook-metadata --identifier key:value format(s).
    out: list[str] = []
    if isinstance(identifiers, dict):
        for k, v in identifiers.items():
            if v is None:
                continue
            s = str(v).strip()
            if not s:
                continue
            out.append(f"{k}:{s}")
    return out


def update_one(book: dict[str, Any], workdir: str) -> bool:
    book_id = book.get("id")
    title = (book.get("title") or "").strip()
    authors = book.get(
        "authors"
    )  # can be list or string depending on calibre version/field
    isbn = (book.get("isbn") or "").strip()
    identifiers = normalize_identifiers(book.get("identifiers"))

    # Make stable, per-book temp paths
    opf_path = os.path.join(workdir, f"{book_id}.opf")
    cover_path = os.path.join(workdir, f"{book_id}.cover.jpg")

    # Build fetch-ebook-metadata command
    fetch_cmd = ["fetch-ebook-metadata", "--opf", opf_path, "--cover", cover_path]

    # Prefer ISBN if available (best matching)
    if isbn:
        fetch_cmd += ["--isbn", isbn]
    else:
        # Otherwise pass identifiers (asin, goodreads, etc.) if present
        for ident in identifiers:
            fetch_cmd += ["--identifier", ident]

        # Fallback: title/authors if we don't have strong identifiers
        if title:
            fetch_cmd += ["--title", title]
        # authors might be list or string
        if authors:
            if isinstance(authors, list):
                fetch_cmd += ["--authors", ", ".join(str(a) for a in authors)]
            else:
                fetch_cmd += ["--authors", str(authors)]

    # Run fetch
    cp_fetch = run(fetch_cmd, capture=True, check=False)
    if cp_fetch.returncode != 0:
        print(
            f"[skip] id={book_id} title={title!r} (fetch failed: {cp_fetch.returncode})",
            file=sys.stderr,
        )
        if cp_fetch.stderr:
            print(cp_fetch.stderr, file=sys.stderr)
        return False

    # Apply OPF to Calibre database
    cp_set = run(
        ["calibredb", "--with-library", LIB, "set_metadata", str(book_id), opf_path],
        capture=True,
        check=False,
    )
    if cp_set.returncode != 0:
        print(
            f"[skip] id={book_id} title={title!r} (set_metadata failed: {cp_set.returncode})",
            file=sys.stderr,
        )
        if cp_set.stderr:
            print(cp_set.stderr, file=sys.stderr)
        return False

    # If we got a cover file, try to set it via --field cover:<path>.
    # (cover is a standard field in calibre's database fields list.) :contentReference[oaicite:4]{index=4}
    if os.path.exists(cover_path) and os.path.getsize(cover_path) > 0:
        cp_cover = run(
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
        if cp_cover.returncode != 0:
            print(
                f"[warn] id={book_id} title={title!r} (cover set failed; continuing)",
                file=sys.stderr,
            )
            if cp_cover.stderr:
                print(cp_cover.stderr, file=sys.stderr)

    print(f"[ok] id={book_id} title={title!r}", file=sys.stderr)
    return True


def main() -> int:
    require_tool("calibredb")
    require_tool("fetch-ebook-metadata")

    books = get_books()
    print(f"[info] Found {len(books)} English EPUB books in: {LIB}", file=sys.stderr)

    ok = 0
    fail = 0

    with tempfile.TemporaryDirectory(prefix="calibre-meta-") as workdir:
        for b in books:
            try:
                if update_one(b, workdir):
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                book_id = b.get("id")
                title = b.get("title")
                print(
                    f"[skip] id={book_id} title={title!r} (exception: {e})",
                    file=sys.stderr,
                )
                fail += 1

    print(f"\n[done] ok={ok} skipped_or_failed={fail}", file=sys.stderr)

    # Optional: if you want the updated Calibre DB metadata embedded into the EPUB files stored in the library,
    # you can run calibredb embed_metadata afterward. :contentReference[oaicite:5]{index=5}
    # (Not done automatically here.)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
