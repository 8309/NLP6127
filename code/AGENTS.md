# Oil Evidence implementation rules

- Use `.venv/Scripts/python.exe` or `uv run --locked`; base launcher is C:/Users/Jiang/.local/bin/python.exe. Never use a runtime from another project.
- Track execution and blocked acceptance items in TASKS.md; distinguish implemented, tested, real-data verified, and human-dependent.
- Fixed initial corpus window: 2018-01-01 through 2026-10-05. Publication and event dates are distinct. Unknown publication dates fail closed in as-of queries; month-only dates use conservative month-end upper bounds.
- Primary collection: EIA STEO, EIA Today in Energy oil-related archive entries, OPEC MOMR when accessible, OFAC Iran/Russia/Venezuela actions. Scope query terms: oil, crude, petroleum, OPEC, refinery, tanker, barrels, sanctions, Hormuz, Suez, Red Sea.
- One request per second per host, honor robots.txt; 25 MB document cap, 300 MB per collection invocation cap. Retry transient network errors at most 3 attempts. Record 403 and do not circumvent access controls.
- Source schema: url, source, title, published_date, available_by, date_precision, date_source, event_date, event_status, redistribution. Passage schema adds document_id, passage_id, page, text, regions, word_count. Labels and answers never enter corpus ingestion.
- No paid model calls without configured service, user-provided budget and positive rate estimates. No secrets in files or logs.
- Model agreement is not human agreement. Keep model-only, reviewed, and independent-human provenance. Formal acceptance does not follow from a smoke test.
- Preserve raw documents and audit trails; do not overwrite a frozen evaluation version or existing pilot queue.

## English publication policy

- Keep public documentation, UI source text, comments, examples and commit messages in English. Preserve archive-compatible schema keys; do not encode untranslated prose to hide it from a language scan.
- Imported sources, historical bilingual labels and frozen evaluation inputs belong in ignored local data, not the public Git tree.
- Run `scripts/check_public_language.py` and the relevant tests before publication. Historical private receipts bind their original implementation, not a later language-translated edition.
- Never rewrite previous commit history or rebind frozen experiment hashes merely to satisfy the public language policy.
