"""Incremental expansion guards, using isolated data only and no network calls."""
import importlib.util
import json
from pathlib import Path

import pytest

from oil_evidence.store import digest, read_jsonl, words, write_json, write_jsonl


def _passage(identifier, document, text):
    return dict(passage_id=identifier, document_id=document, source="EIA",
                title="Crude oil report", url="https://example.test/" + document,
                published_date="2023-01-01", available_by="2023-01-01",
                date_precision="day", date_source="official archive", event_date=None,
                event_status="unlabelled", redistribution="eia_attribution", page=1,
                regions=[], region_method="keyword_heuristic", text=text,
                word_count=len(words(text)))


@pytest.fixture
def isolated_expansion(tmp_path, monkeypatch):
    import oil_evidence.corpus as corpus

    script = Path(__file__).parents[1] / "scripts/expand_corpus.py"
    spec = importlib.util.spec_from_file_location("isolated_expansion_v2", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "project/data"
    data.mkdir(parents=True)
    monkeypatch.setattr(module, "DATA", data)
    monkeypatch.setattr(module, "ROOT", data / "expansion_runs/test-v2")
    monkeypatch.setattr(corpus, "DATA", data)
    monkeypatch.setattr(corpus, "config", lambda: {"corpus": dict(
        start_date="2018-01-01", end_date="2026-10-05", chunk_words=200,
        min_words=65, request_delay_seconds=1, max_document_mb=25,
        download_budget_mb=300)})

    original = _passage("original-passage", "original-document",
        "Crude oil production in the original country remained stable during the reporting period. " * 8)
    previous = _passage("previous-expansion", "previous-document",
        "Petroleum exports through the previous terminal increased following the pipeline reopening. " * 8)
    write_jsonl(data / "passages.jsonl", [original, previous])
    documents, sources = [], []
    for identifier in ["original-document", "previous-document", "old-zero-contribution"]:
        # The old zero-contribution file would now yield oil passages if reprocessed.
        text = "Crude oil transport in this archived country depended on the pipeline and export terminal. " * 12
        raw = ("<article>" + identifier + " " + text + "</article>").encode()
        raw_path = data / "raw" / (identifier + ".html")
        raw_path.parent.mkdir(exist_ok=True)
        raw_path.write_bytes(raw)
        source = dict(url="https://example.test/" + identifier, source="EIA",
                      title="Crude oil report " + identifier, published_date="2023-01-01",
                      available_by="2023-01-01", date_precision="day", date_source="archive",
                      event_date=None, event_status="unlabelled", redistribution="eia_attribution")
        sources.append(source)
        documents.append(dict(source, document_id=identifier,
                              raw_path=raw_path.relative_to(data).as_posix(),
                              raw_sha256=digest(raw), bytes=len(raw)))
    write_jsonl(data / "documents.jsonl", documents)
    write_jsonl(data / "sources.jsonl", sources)

    annotation_path = data / "expansion_runs/test-v1/baseline/passages.jsonl"
    write_jsonl(annotation_path, [original])
    write_json(data / "reports/passage_annotation.json", dict(
        corpus_sha256=digest(annotation_path.read_bytes()), labels_sha256="fixture"))
    write_json(data / "reports/corpus_expansion.json", dict(
        snapshot_root=str(annotation_path.parent.parent),
        corpus_sha256=digest((data / "passages.jsonl").read_bytes())))
    write_json(data / "reports/corpus.json", dict(passages=2))
    for name in ["codex", "passage_labels", "passage_review"]:
        write_jsonl(data / f"annotations/{name}.jsonl", [dict(fixture=name)])
    for name in ["config.toml", "uv.lock", "pyproject.toml"]:
        (data.parent / name).write_text("# isolated fixture\n", encoding="utf-8")
    return module, data, annotation_path


def test_v2_snapshot_separates_run_and_annotation_baselines(isolated_expansion):
    module, data, annotation_path = isolated_expansion
    original_annotation_bytes = annotation_path.read_bytes()
    before = (data / "passages.jsonl").read_bytes()
    manifest = module.snapshot()
    assert manifest["passages"] == 2
    assert manifest["hashes"]["passages.jsonl"] == digest(before)
    assert manifest["annotation_baseline_sha256"] == digest(original_annotation_bytes)
    assert manifest["annotation_baseline_sha256"] != manifest["hashes"]["passages.jsonl"]
    assert Path(manifest["annotation_baseline_path"]) == annotation_path
    assert (module.ROOT / "baseline/passages.jsonl").read_bytes() == before
    assert annotation_path.read_bytes() == original_annotation_bytes
    assert module.snapshot() == manifest
    for name in ["config.toml", "uv.lock", "pyproject.toml"]:
        assert manifest["hashes"]["project/" + name] == digest((data.parent / name).read_bytes())


def test_v2_snapshot_detects_frozen_backup_changes(isolated_expansion):
    module, _, _ = isolated_expansion
    module.snapshot()
    target = module.ROOT / "baseline/passages.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Baseline backup changed"):
        module.snapshot()


def test_snapshot_failure_cleans_scratch_and_can_resume(isolated_expansion, monkeypatch):
    module, data, _ = isolated_expansion
    before = (data / "passages.jsonl").read_bytes()
    copy = module.shutil.copy2

    def fail_late(source, destination, *args, **kwargs):
        if Path(source).name == "uv.lock":
            raise OSError("fixture snapshot copy failure")
        return copy(source, destination, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(module.shutil, "copy2", fail_late)
        with pytest.raises(OSError, match="fixture snapshot copy failure"):
            module.snapshot()
    assert not (module.ROOT / "baseline").exists()
    assert not list(module.ROOT.glob(".baseline-*"))
    assert (data / "passages.jsonl").read_bytes() == before
    assert module.snapshot()["passages"] == 2


def test_initial_annotation_snapshot_records_final_not_scratch_path(isolated_expansion):
    module, data, _ = isolated_expansion
    report_path = data / "reports/passage_annotation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["corpus_sha256"] = digest((data / "passages.jsonl").read_bytes())
    write_json(report_path, report)
    (data / "reports/corpus_expansion.json").unlink()
    manifest = module.snapshot()
    assert Path(manifest["annotation_baseline_path"]) == module.ROOT / "baseline/passages.jsonl"
    assert Path(manifest["annotation_baseline_path"]).exists()
    assert module.snapshot() == manifest


def test_snapshot_resume_checks_separate_annotation_baseline(isolated_expansion):
    module, _, annotation_path = isolated_expansion
    module.snapshot()
    annotation_path.write_bytes(annotation_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Original annotation baseline changed"):
        module.snapshot()


@pytest.mark.parametrize("manifest_name,key", [
    ("documents.jsonl", "available_by"), ("sources.jsonl", "title")])
def test_old_source_metadata_cannot_change(isolated_expansion, manifest_name, key):
    module, data, _ = isolated_expansion
    module.snapshot()
    module.verify_baseline_sources()
    rows = read_jsonl(data / manifest_name)
    rows[0][key] = "modified"
    write_jsonl(data / manifest_name, rows)
    with pytest.raises(ValueError, match="Baseline metadata changed"):
        module.verify_baseline_sources()


@pytest.mark.parametrize("manifest_name", ["documents.jsonl", "sources.jsonl"])
def test_old_source_metadata_cannot_be_removed(isolated_expansion, manifest_name):
    module, data, _ = isolated_expansion
    module.snapshot()
    write_jsonl(data / manifest_name, read_jsonl(data / manifest_name)[1:])
    with pytest.raises(ValueError, match="Baseline metadata changed"):
        module.verify_baseline_sources()


def test_baseline_raw_integrity_is_checked(isolated_expansion):
    module, data, _ = isolated_expansion
    module.snapshot()
    row = read_jsonl(data / "documents.jsonl")[0]
    path = data / row["raw_path"]
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="Baseline raw file changed"):
        module.verify_baseline_sources()


def test_append_only_new_manifests_preserve_full_staged_prefix(isolated_expansion):
    module, data, annotation_path = isolated_expansion
    baseline = read_jsonl(data / "passages.jsonl")
    annotation_bytes = annotation_path.read_bytes()
    annotation_hashes = {name: digest((data / f"annotations/{name}.jsonl").read_bytes())
                         for name in ["codex", "passage_labels", "passage_review"]}
    module.snapshot()
    text = "Crude oil production from the newly discovered offshore field increased after the export pipeline opened. " * 12
    raw = ("<article>" + text + "</article>").encode()
    (data / "raw/new.html").write_bytes(raw)
    source = dict(url="https://example.test/new", source="EIA", title="New oil field",
                  available_by="2024-01-01", published_date="2024-01-01",
                  date_precision="day", date_source="archive")
    write_jsonl(data / "sources.jsonl", read_jsonl(data / "sources.jsonl") + [source])
    write_jsonl(data / "documents.jsonl", read_jsonl(data / "documents.jsonl") + [dict(
        source, document_id="new-document", raw_path="raw/new.html", raw_sha256=digest(raw))])
    module.verify_baseline_sources()
    report = module.stage()
    staged = read_jsonl(module.ROOT / "stage/passages.jsonl")
    assert staged[:len(baseline)] == baseline
    assert report["retained_baseline_passages"] == 2
    assert report["new_passages"] > 0
    assert {p["document_id"] for p in staged[len(baseline):]} == {"new-document"}
    assert not report["extraction_errors"]
    assert read_jsonl(data / "passages.jsonl") == baseline
    assert annotation_path.read_bytes() == annotation_bytes
    assert all(digest((data / f"annotations/{name}.jsonl").read_bytes()) == expected
               for name, expected in annotation_hashes.items())


@pytest.mark.parametrize("damage", ["corrupt", "missing"])
def test_existing_raw_damage_never_refetches_or_overwrites(isolated_expansion, monkeypatch, damage):
    import oil_evidence.corpus as corpus

    _, data, _ = isolated_expansion
    row = read_jsonl(data / "documents.jsonl")[0]
    path = data / row["raw_path"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupted fixture")
    before = (data / "documents.jsonl").read_bytes()
    requests = []
    closes = []

    class NoNetworkFetcher:
        bytes = 0

        def get(self, url):
            requests.append(url)
            raise AssertionError("Existing raw failure must not issue a request")

        def close(self):
            closes.append(True)

    monkeypatch.setattr(corpus, "Fetcher", NoNetworkFetcher)
    with pytest.raises(ValueError, match="refuse refetch/overwrite"):
        corpus.collect(limit=1)
    assert not requests
    assert closes == [True]
    assert (data / "documents.jsonl").read_bytes() == before
    assert (not path.exists()) if damage == "missing" else path.read_bytes() == b"corrupted fixture"


@pytest.mark.parametrize("unsafe", ["../elsewhere", "run/path", "run\\path", "D:\\other", "",
                                     "ab", "UPPERCASE", "a" * 82, ".hidden", "run name"])
def test_run_selection_rejects_unsafe_paths(isolated_expansion, unsafe):
    module, _, _ = isolated_expansion
    before = module.ROOT
    with pytest.raises(ValueError, match="Invalid expansion run ID"):
        module.select_run(unsafe)
    assert module.ROOT == before


@pytest.mark.parametrize("device", ["con", "prn", "aux", "nul", "com1", "com9", "lpt1", "lpt9"])
def test_run_selection_rejects_windows_device_names(isolated_expansion, device):
    module, _, _ = isolated_expansion
    before = module.ROOT
    with pytest.raises(ValueError, match="Invalid expansion run ID"):
        module.select_run(device)
    assert module.ROOT == before


def test_run_selection_stays_within_data_expansion_directory(isolated_expansion):
    module, data, _ = isolated_expansion
    assert module.select_run("official-2018-2026-v2") == data / "expansion_runs/official-2018-2026-v2"


def test_manifest_writer_lock_is_exclusive_and_released(isolated_expansion):
    module, data, _ = isolated_expansion
    lock = data / ".corpus_writer.lock"
    with module.pipeline_lock():
        metadata = json.loads(lock.read_text(encoding="utf-8"))
        assert metadata["run"] == str(module.ROOT) and metadata["pid"] > 0
        with pytest.raises(RuntimeError, match="Another pipeline writer"):
            with module.pipeline_lock():
                pytest.fail("Nested writer must not acquire the lock")
        assert lock.exists()
    assert not lock.exists()
    with pytest.raises(RuntimeError, match="fixture error"):
        with module.pipeline_lock():
            raise RuntimeError("fixture error")
    assert not lock.exists()


def test_foreign_or_stale_lock_is_preserved(isolated_expansion):
    module, data, _ = isolated_expansion
    lock = data / ".corpus_writer.lock"
    lock.write_text('{"pid":0,"run":"foreign"}', encoding="utf-8")
    before = lock.read_bytes()
    with pytest.raises(RuntimeError, match="Another pipeline writer"):
        with module.pipeline_lock():
            pytest.fail("An existing lock must fail closed")
    assert lock.read_bytes() == before


def test_collector_always_closes_on_manifest_error(isolated_expansion, monkeypatch):
    import oil_evidence.corpus as corpus

    _, data, _ = isolated_expansion
    (data / "documents.jsonl").write_text("invalid json\n", encoding="utf-8")
    closes = []

    class NoNetworkFetcher:
        def get(self, url):
            pytest.fail("Manifest parsing failure must not issue a request")

        def close(self):
            closes.append(True)

    monkeypatch.setattr(corpus, "Fetcher", NoNetworkFetcher)
    with pytest.raises(json.JSONDecodeError):
        corpus.collect()
    assert closes == [True]


@pytest.mark.parametrize("outcome", ["cached", "success", "failure", "budget"])
def test_collector_closes_on_every_regular_exit(isolated_expansion, monkeypatch, outcome):
    import oil_evidence.corpus as corpus

    _, data, _ = isolated_expansion
    closes, requests = [], []
    if outcome != "cached":
        write_jsonl(data / "documents.jsonl", [])
    before = (data / "documents.jsonl").read_bytes()

    class FakeFetcher:
        bytes = 0

        def get(self, url):
            requests.append(url)
            if outcome == "failure":
                raise RuntimeError("fixture collection failure")
            if outcome == "budget":
                raise ValueError("Download budget reached")
            raw = b"<article>Crude oil report.</article>"
            self.bytes += len(raw)
            return raw, url, "text/html"

        def close(self):
            closes.append(True)

    monkeypatch.setattr(corpus, "Fetcher", FakeFetcher)
    report = corpus.collect(limit=1)
    assert closes == [True]
    assert report["added"] == (1 if outcome == "success" else 0)
    assert bool(requests) == (outcome != "cached")
    if outcome in {"cached", "failure", "budget"}:
        assert (data / "documents.jsonl").read_bytes() == before


def test_publish_refuses_changed_frozen_run_manifest(isolated_expansion):
    import numpy as np

    module, data, _ = isolated_expansion
    manifest = module.snapshot()
    staged = module.ROOT / "stage"
    rows = read_jsonl(data / "passages.jsonl") + [_passage(
        "new-published-passage", "new-published-document", "New crude oil evidence. " * 30)]
    write_jsonl(staged / "passages.jsonl", rows)
    corpus_hash = digest((staged / "passages.jsonl").read_bytes())
    write_json(staged / "reports/corpus.json", dict(passages=len(rows), new_passages=1,
        corpus_sha256=corpus_hash, baseline_sha256=manifest["hashes"]["passages.jsonl"],
        labels_coverage=1, meets_size_requirement=False))
    write_json(staged / "reports/expansion_quality.json", dict(corpus_sha256=corpus_hash,
        new_passages_page_provenance_passed=True,
        documents_sha256=digest((data / "documents.jsonl").read_bytes()),
        sources_sha256=digest((data / "sources.jsonl").read_bytes()),
        implementation_hashes=module.implementation_hashes()))
    (staged / "index").mkdir()
    np.save(staged / "index/vectors.npy", np.ones((len(rows), 2)), allow_pickle=False)
    write_json(staged / "index/manifest.json", dict(corpus_sha256=corpus_hash,
        vectors_sha256=digest((staged / "index/vectors.npy").read_bytes()),
        passage_ids=[p["passage_id"] for p in rows], dimension=2))
    write_jsonl(staged / "reports/near_duplicate_candidates.jsonl", [])
    write_jsonl(staged / "reports/source_qa_samples.jsonl", [])
    frozen_manifest = module.ROOT / "published" / corpus_hash / "run_baseline_manifest.json"
    write_json(frozen_manifest, dict(corrupt_fixture=True))
    frozen_before = frozen_manifest.read_bytes()
    live_before = (data / "passages.jsonl").read_bytes()
    with pytest.raises(ValueError, match="Published run baseline manifest changed"):
        module.publish()
    assert frozen_manifest.read_bytes() == frozen_before
    assert (data / "passages.jsonl").read_bytes() == live_before
