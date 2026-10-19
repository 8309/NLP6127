# Oil Evidence Teacher Demo

This delivery contains the complete active retrieval corpus, its matching vector index, the local query embedding model, and a portable demonstration interface. You can compare keyword, vector and hybrid retrieval, apply a publication cutoff and region filter, inspect exact source pages, and browse the 1,000 test labels without an API key.

## Start on Windows

1. Copy the complete `oil-evidence-teacher` folder to a writable location. Keep its subfolders together.
2. Install Python 3.12 if it is not already available.
3. Double-click `SETUP.cmd`. The setup downloads Python dependencies into this folder's `.venv` and runs an integrity and retrieval check. It does not install packages globally.
4. Double-click `RUN.cmd` and open `http://127.0.0.1:8501`. Keep the terminal open while using the interface.

The first setup needs internet access for Python packages. Subsequent keyword, vector and hybrid searches use the bundled model and index locally; they do not require model downloads, an API key, a GPU or a generation account. The Python runtime and installed dependency wheels are not bundled.

## Start from a terminal

Run these commands from the project folder on Windows:

```powershell
py -3.12 setup_teacher.py
.\.venv\Scripts\python.exe run_teacher.py --check
.\.venv\Scripts\python.exe run_teacher.py
```

On Linux or macOS, use `python3.12 setup_teacher.py`, then `.venv/bin/python run_teacher.py`. The dependency installer and launcher support these paths, but execution has been checked on Windows only. If port 8501 is busy, use `run_teacher.py --port 8502`.

For a command-line query without the interface:

```powershell
.\.venv\Scripts\python.exe run_teacher.py --query "How do hurricanes affect Gulf of Mexico crude oil production?" --mode hybrid --as-of 2026-10-05
```

## What to inspect

Begin with the default English question, compare `bm25`, `dense` and `hybrid`, and move the cutoff date back to see which sources remain eligible. Each result includes a passage ID, publication availability date, source link, page number and the included page text. The Test labels tab shows the full question and reference labels for each test record.

The bundled library contains 11,815 passages from 1,527 contributing documents, 5,703 extracted source pages, and a float32 vector matrix of shape 11,815 by 384. `data/corpus/documents.jsonl` retains metadata for all 1,990 registered documents, including documents without retained passages. The model is the same local ONNX implementation used to build the index; rebuilding document vectors is unnecessary. Keyword rankings are built in memory from the corpus at startup.

## Answers and generation

The default Offline source excerpts mode shows retrieved quotations. It is not a synthesized answer and does not certify answerability. This is the complete no-account demonstration path.

Optional Codex answer mode uses your own installed and authenticated Codex CLI. Selecting it sends retrieved evidence to your configured service and consumes your own allowance. It reuses the preserved project generation and support-check implementation; no credentials or original execution logs are included. Optional new generation creates new query receipts inside the project folder. Automated delivery testing does not invoke this account-dependent path.

The portable loader validates this delivery's file manifest and source-page offsets. It is a new handoff adapter, not the historical staged launcher, and does not claim to reproduce historical calls or establish new evaluation scores. The preserved original implementation is under `code/`.

## Contents and scope

`report.pdf` and `presentation.pptx` cover the 1,000 human-annotated TEST records in `data/test_labels.jsonl`. Results are expanded under `results/`. Cover identifiers are intentionally blank. See `data/ANNOTATION_PROVENANCE.json`, `data/LABEL_ORIGIN.json` and `results/human_agreement.json`. Questions, labels and TEST evaluation scores are unchanged. The two independent raw human annotation streams are not bundled. Answer generation and answer-quality judging remain automatic.

Raw downloaded PDF and HTML files, historical run logs, the embedding cache database, virtual environments and account credentials are excluded. Model attribution and the upstream MIT notice are in `MODEL_NOTICES.md`. Corpus source attribution and redistribution metadata remain attached to each passage and document; preparing this local package is not blanket clearance to publish its source texts.

`TEACHER_MANIFEST.json` lists payload sizes and SHA-256 hashes. The `--check` command validates them and exercises all three retrieval modes and a cutoff with no eligible evidence. If it reports a mismatch, restore a fresh copy of the complete folder rather than modifying the index manifest.
