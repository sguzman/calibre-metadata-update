Calibre Metadata Update
=======================

Bulk metadata updater for Calibre EPUB books with idempotent processing.

Intent
------
- Iterate through a Calibre library and update metadata for EPUB books.
- Prefer books that are English or missing language.
- Fetch richer metadata when current data is incomplete.
- Embed metadata directly into EPUB files after updating the Calibre DB.
- Avoid reprocessing the same book on subsequent runs by default.

Structure
---------
Main flow in `main.py`:
- **Candidate selection**: `calibredb list --for-machine` with EPUB prefilter, then filter to English-or-missing-language.
- **Snapshot + scoring**: compute a stable hash of current DB metadata and decide if the record is "good enough".
- **Two paths**:
  - **Embed-only** when metadata is already good enough.
  - **Fetch → apply → embed** when metadata needs improvement.
- **State tracking**: a JSON file records each book id and last processing status to ensure idempotence.

Behavior
--------
- **Idempotent by default**: each book is processed only once (per book id).  
  This is controlled by `REPROCESS_ON_METADATA_CHANGE` in `main.py`.
  - `False` (default): skip any book already processed successfully.
  - `True`: reprocess a book when its metadata snapshot hash changes.
- **Failure handling**: if a step fails, the error is recorded and the script continues.
- **Embedding**: successful runs embed metadata into EPUB files via `calibredb embed_metadata`.

Configuration
-------------
Edit in `main.py`:
- `DEFAULT_LIB`: default Calibre library path.
- `REPROCESS_ON_METADATA_CHANGE`: reprocess on metadata changes or not.
- `MIN_SCORE_TO_SKIP_FETCH`: how strict “good enough” is.
- `ENGLISH_CODES` / `INCLUDE_MISSING_LANGUAGE`: language filtering.
- `DELAY_BETWEEN_FETCHES_SECONDS`: throttle external metadata fetching.

Usage
-----
Requires `calibredb` and `fetch-ebook-metadata` on your PATH.

Run against the default library:
```bash
python main.py
```

Run against a specific library path:
```bash
python main.py --library "/path/to/Calibre Library"
```

State file
----------
The state is stored in the library directory as:
```
.calibre_metadata_state.json
```
It tracks per-book status, last hash, timestamps, and failure counts.
