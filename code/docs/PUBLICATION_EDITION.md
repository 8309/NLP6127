# English publication edition

**Annotation provenance corrected on 19 October 2026.** The delivery contains 1,000 TEST records with human reference annotations. Earlier machine-origin descriptions of these labels were incorrect. Automatic answer generation and judging, older pilot records and machine source checks retain their own provenance. Questions, label values and TEST evaluation scores are unchanged. Overall raw human annotation agreement is 91.13% for TEST. Per-field agreement and kappa values are not reported. Historical filenames, schema keys and hashes are compatibility references.


The public repository is an English presentation edition of the oil-evidence project. Documentation, UI source text, comments and examples are English. Historical language-specific schema keys, such as `question_zh` and `summary_zh`, remain for archive compatibility; the public seed examples contain English values. Ignored local source documents and historical bilingual records are not rewritten or uploaded.

## Historical evidence and current code

The 11,815-passage staged corpus, its 384-dimensional index, 24 operational retrieval checks and 50 initial development questions were verified in the preserved canonical local implementation. The private source-closeout receipts bind 19 original implementation files; the development-question snapshot binds 71 inputs. Translating UI or preview text changes implementation hashes even when control flow is unchanged. Those historical receipts therefore do not retroactively certify the translated public edition.

Exact historical reconstruction needs the matching canonical source snapshot and the excluded original data, caches, labels and model files. A clean public clone can install dependencies and run temporary-fixture tests, but it cannot reproduce historical corpus results from an empty data directory. Copying old receipts into a different implementation is not a valid replay; the existing fail-closed guards remain in place.

English-edition test and language-scan results are recorded separately in [reports/english_publication.json](../reports/english_publication.json). Existing corpus and development records are development artifacts, not human gold, independent-human agreement or formal course acceptance. The old retrieval regression and remaining evaluation work are preserved in [current progress](GITHUB_PROGRESS.md).

## Publication checks

```powershell
uv sync --locked --group dev
uv run --locked python scripts/check_public_language.py
uv run --locked python -m pytest -q
```

The language checker examines current Git-visible UTF-8 files and names, not ignored local data or previous commits. A lightweight GitHub Actions workflow runs the same language check on pushes and pull requests. Existing Git history is retained without force-pushing or rewriting earlier commits.

Subsequent public changes should be made in this English edition and checked before publication. Changes to the separately preserved canonical experiment must remain versioned and hash-audited; do not rebind immutable experiment receipts solely to match translated files.
