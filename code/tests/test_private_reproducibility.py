"""Synthetic local-private packaging fixtures; never actual coursework data."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import zipfile

import pytest


@pytest.fixture
def bundle():
    spec = importlib.util.spec_from_file_location("private_bundle", Path(__file__).parents[1] / "scripts/package_private_reproducibility.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def snapshot(bundle, tmp_path):
    canonical, publication = tmp_path / "canonical", tmp_path / "publication"
    publication.mkdir()
    run = canonical / "data/machine_runs/formal-experiments/fixture"
    stage = canonical / "data/unified_runs/fixture/stage"
    frozen = canonical / "data/machine_runs/frozen_evaluation/fixture"
    sample = canonical / "data/machine_runs/answer-samples/fixture"
    retrieval = canonical / "data/machine_runs/candidate-retrieval/fixture"
    runtime = canonical / "data/machine_runs/experiment-runtime/fixture"
    audit = canonical / "data/machine_runs/engineering-checkpoints/fixture/audit.json"

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
        return path

    raw = write(canonical / "data/raw/source.pdf", b"synthetic official raw source")
    doc = dict(document_id="source", raw_path="raw/source.pdf", raw_sha256=bundle.sha(raw), bytes=raw.stat().st_size,
               redistribution="review_required", source="fixture", url="https://example.invalid/source")
    write(stage / "documents.jsonl", json.dumps(doc).encode() + b"\n")
    model = write(canonical / "models/model.onnx", b"synthetic model bytes")
    write(stage / "index/manifest.json", dict(onnx_sha256={"model.onnx": bundle.sha(model)}))
    write(stage / "index/vectors.npy", b"synthetic vectors")
    write(frozen / "manifest.json", dict(version="fixture"))
    write(sample / "sample_plan.json", dict(sample_seed=6127))
    write(retrieval / "rankings.jsonl", b"{}\n")
    write(runtime / "implementation.artifact", b"synthetic frozen source")
    protected = {str(p): bundle.sha(p) for p in (stage / "documents.jsonl", stage / "index/manifest.json",
        frozen / "manifest.json", sample / "sample_plan.json", model)}
    ids = [f"q-{i}" for i in range(1000)]
    plan = dict(protected_sha256=dict(protected), all_test_record_ids=ids, answer_record_ids=ids[:100],
        retrieval=dict(run=str(retrieval)), dataset_version="fixture", answer_sample_plan_sha256=bundle.sha(sample / "sample_plan.json"))
    write(run / "plan.json", plan)
    group = write(run / "groups/000000.json", dict(record_ids=ids[:100],
        rows=[dict(record_id=qid, variant=variant) for qid in ids[:100] for variant in sorted(bundle.VARIANTS)]))
    result = dict(status="completed_formal_machine_only_experiment", human_gold=False, formal_heldout_test=True,
        completed_retrieval_questions=1000, completed_answer_questions=100, completed_variant_answers=600,
        original_frozen_inputs_unchanged=True, error=None, dataset_version="fixture",
        answer_sample_plan_sha256=plan["answer_sample_plan_sha256"], summary={k:dict(questions=100) for k in bundle.VARIANTS},
        paired_automatic_correctness={}, usage=dict(unknown_attempts=3), group_artifacts_sha256={"groups/000000.json":bundle.sha(group)})
    write(run / "result.json", result)
    protected.update({str(group):bundle.sha(group), str(run / "plan.json"):bundle.sha(run / "plan.json")})
    receipt = dict(status="completed_formal_result_offline_audit_passed", human_gold=False, formal_heldout_test=True,
        native_calls=0, answer_questions=100, variant_answers=600, ranked_test_questions=1000, ranked_query_conditions=6000,
        result_sha256=bundle.sha(run / "result.json"), plan_sha256=bundle.sha(run / "plan.json"), protected_sha256=protected,
        summary=result["summary"], paired_automatic_correctness={}, usage=result["usage"])
    write(audit, receipt)
    args = dict(canonical_root=canonical, formal_run=run, audit_path=audit, audit_sha256=bundle.sha(audit),
        runtime_dir=runtime, publication_root=publication)
    return dict(args=args, write=write, canonical=canonical, publication=publication, run=run, stage=stage,
                audit=audit, receipt=receipt, plan=plan, result=result, raw=raw, model=model, group=group)


def test_inventory_preserves_original_mapping_and_rights(bundle, snapshot):
    value = bundle.inventory(**snapshot["args"])
    assert value["raw_document_count"] == 1
    assert value["rights_review_required"] and not value["cloud_sharing_authorized"]
    assert not value["model_license_file_present"] and not value["frozen_files_rewritten"]
    assert value["entries"]["canonical/data/raw/source.pdf"]["original_absolute_path"] == str(snapshot["raw"])
    assert value["source_license_counts"] == {"review_required": 1}


@pytest.mark.parametrize("field,value", [("status","partial"), ("completed_answer_questions",99),
    ("completed_variant_answers",599), ("completed_retrieval_questions",999), ("human_gold",True),
    ("error","failed"), ("original_frozen_inputs_unchanged",False), ("completed_answer_questions",100.0)])
def test_incomplete_or_incompatible_results_refused(bundle, snapshot, field, value):
    snapshot["result"][field] = value
    snapshot["write"](snapshot["run"] / "result.json", snapshot["result"])
    with pytest.raises(ValueError, match="100/600"):
        bundle.inventory(**snapshot["args"])


@pytest.mark.parametrize("field,value", [("status","fixture_passed"), ("native_calls",1), ("human_gold",True),
    ("variant_answers",599), ("result_sha256","0"*64), ("summary",{}), ("usage",{})])
def test_audit_provenance_or_binding_refused(bundle, snapshot, field, value):
    snapshot["receipt"][field] = value
    snapshot["write"](snapshot["audit"], snapshot["receipt"])
    snapshot["args"]["audit_sha256"] = bundle.sha(snapshot["audit"])
    with pytest.raises(ValueError):
        bundle.inventory(**snapshot["args"])


def test_unpinned_audit_refused(bundle, snapshot):
    snapshot["args"]["audit_sha256"] = "0"*64
    with pytest.raises(ValueError, match="audit pin"):
        bundle.inventory(**snapshot["args"])


@pytest.mark.parametrize("which", ["raw", "model", "group"])
def test_original_byte_mutations_refused(bundle, snapshot, which):
    snapshot[which].write_bytes(b"changed fixture")
    with pytest.raises(ValueError):
        bundle.inventory(**snapshot["args"])


def test_missing_original_refused(bundle, snapshot):
    snapshot["raw"].unlink()
    with pytest.raises(ValueError):
        bundle.inventory(**snapshot["args"])


@pytest.mark.parametrize("name", [".env", "auth.json", "credentials.json", "nested/.env.local", "token.key"])
def test_forbidden_supplement_refused(bundle, snapshot, name):
    snapshot["write"](snapshot["model"].parent / name, b"private fixture")
    with pytest.raises(ValueError, match="Credential/environment"):
        bundle.inventory(**snapshot["args"])


def test_outside_original_closure_refused(bundle, snapshot, tmp_path):
    outside = snapshot["write"](tmp_path / "outside.txt", b"unrelated")
    snapshot["receipt"]["protected_sha256"][str(outside)] = bundle.sha(outside)
    snapshot["write"](snapshot["audit"], snapshot["receipt"])
    snapshot["args"]["audit_sha256"] = bundle.sha(snapshot["audit"])
    with pytest.raises(ValueError, match="escapes"):
        bundle.inventory(**snapshot["args"])


def test_synthetic_zip_is_exact_and_never_overwrites(bundle, snapshot):
    manifest = bundle.inventory(**snapshot["args"])
    output = snapshot["canonical"] / "data/machine_runs/private_reproducibility/fixture.zip"
    before = {e["original_absolute_path"]:bundle.sha(e["original_absolute_path"]) for e in manifest["entries"].values()}
    result = bundle.package(manifest, snapshot["canonical"], output)
    assert not result["cloud_sharing_authorized"]
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(manifest["entries"]) | {"MANIFEST.json"}
        assert archive.read("canonical/data/raw/source.pdf") == snapshot["raw"].read_bytes()
        assert json.loads(archive.read("MANIFEST.json")) == manifest
    assert all(bundle.sha(p) == h for p,h in before.items())
    with pytest.raises(ValueError, match="destination"):
        bundle.package(manifest, snapshot["canonical"], output)


def test_altered_inventory_refused(bundle, snapshot):
    manifest = bundle.inventory(**snapshot["args"])
    manifest["entries"].pop("canonical/data/raw/source.pdf")
    with pytest.raises(ValueError, match="Inventory"):
        bundle.package(manifest, snapshot["canonical"], snapshot["canonical"] / "data/machine_runs/private_reproducibility/fixture.zip")


def test_secret_bytes_stop_before_final_publication(bundle, snapshot):
    secret = b"sk-" + b"z" * 32
    snapshot["write"](snapshot["model"].parent / "unsafe.txt", b"x" * 262140 + secret)
    manifest = bundle.inventory(**snapshot["args"])
    output = snapshot["canonical"] / "data/machine_runs/private_reproducibility/secret.zip"
    with pytest.raises(ValueError, match="credential") as failure:
        bundle.package(manifest, snapshot["canonical"], output)
    assert secret.decode() not in str(failure.value) and not output.exists()
    assert output.with_suffix(".zip.incomplete").exists()


def test_output_escape_refused(bundle, snapshot, tmp_path):
    manifest = bundle.inventory(**snapshot["args"])
    with pytest.raises(ValueError, match="destination"):
        bundle.package(manifest, snapshot["canonical"], tmp_path / "public.zip")


def test_duplicate_raw_id_refused(bundle, snapshot):
    registry = snapshot["stage"] / "documents.jsonl"
    registry.write_bytes(registry.read_bytes() * 2)
    digest = bundle.sha(registry)
    snapshot["plan"]["protected_sha256"][str(registry)] = digest
    snapshot["write"](snapshot["run"] / "plan.json", snapshot["plan"])
    snapshot["receipt"]["protected_sha256"][str(registry)] = digest
    snapshot["receipt"]["protected_sha256"][str(snapshot["run"] / "plan.json")] = bundle.sha(snapshot["run"] / "plan.json")
    snapshot["receipt"]["plan_sha256"] = bundle.sha(snapshot["run"] / "plan.json")
    snapshot["write"](snapshot["audit"], snapshot["receipt"])
    snapshot["args"]["audit_sha256"] = bundle.sha(snapshot["audit"])
    with pytest.raises(ValueError, match="Duplicate raw"):
        bundle.inventory(**snapshot["args"])


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended-path fixture")
def test_extended_audit_keys_keep_original_aliases_and_deduplicate(bundle, snapshot):
    pins = snapshot["receipt"]["protected_sha256"]
    snapshot["receipt"]["protected_sha256"] = {str(bundle.io_path(p)): h for p, h in pins.items()}
    snapshot["write"](snapshot["audit"], snapshot["receipt"])
    snapshot["args"]["audit_sha256"] = bundle.sha(snapshot["audit"])
    value = bundle.inventory(**snapshot["args"])
    entry = value["entries"]["canonical/models/model.onnx"]
    assert str(bundle.io_path(snapshot["model"])) in entry["original_path_aliases"]
    assert str(snapshot["model"]) in entry["original_path_aliases"]
    assert sum(name.lower().endswith("models/model.onnx") for name in value["entries"]) == 1
    assert entry["original_absolute_path"].startswith("\\\\?\\")


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended-path fixture")
def test_conflicting_extended_alias_pin_refused(bundle, snapshot):
    snapshot["receipt"]["protected_sha256"][str(bundle.io_path(snapshot["model"]))] = "0" * 64
    snapshot["write"](snapshot["audit"], snapshot["receipt"])
    snapshot["args"]["audit_sha256"] = bundle.sha(snapshot["audit"])
    with pytest.raises(ValueError, match="Conflicting path-alias"):
        bundle.inventory(**snapshot["args"])


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended-path fixture")
def test_extended_project_roots_have_same_logical_containment(bundle, snapshot):
    normal = bundle.inventory(**snapshot["args"])
    for name in ("canonical_root", "formal_run", "audit_path", "runtime_dir", "publication_root"):
        snapshot["args"][name] = bundle.io_path(snapshot["args"][name])
    assert bundle.inventory(**snapshot["args"]) == normal


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended-path fixture")
def test_long_native_receipt_and_output_use_extended_io(bundle, snapshot):
    deep = snapshot["run"] / "jobs" / ("native-" + "a" * 40)
    for index in range(7):
        deep /= f"attempt-{index}-" + "b" * 30
    native = deep / "receipt.json"
    bundle.io_path(native.parent).mkdir(parents=True)
    bundle.io_path(native).write_bytes(b'{"fixture":"long native receipt"}')
    assert len(str(native)) > 260
    snapshot["receipt"]["protected_sha256"][str(bundle.io_path(native))] = bundle.sha(native)
    snapshot["write"](snapshot["audit"], snapshot["receipt"])
    snapshot["args"]["audit_sha256"] = bundle.sha(snapshot["audit"])
    value = bundle.inventory(**snapshot["args"])
    name = "canonical/" + native.relative_to(snapshot["canonical"]).as_posix()
    assert name in value["entries"]
    assert bundle.safe_file(bundle.io_path(native), {"canonical":snapshot["canonical"]}) == name
    output = snapshot["canonical"] / "data/machine_runs/private_reproducibility"
    for index in range(6):
        output /= f"output-{index}-" + "c" * 30
    output /= "fixture.zip"
    bundle.package(value, snapshot["canonical"], bundle.io_path(output))
    with zipfile.ZipFile(bundle.io_path(output)) as archive:
        assert archive.read(name) == bundle.io_path(native).read_bytes()
    assert not bundle.io_path(output.with_suffix(".zip.incomplete")).exists()


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended-path fixture")
@pytest.mark.parametrize("name", ["\\\\?\\GLOBALROOT\\Device\\HarddiskVolume1\\secret",
    "\\\\.\\C:\\secret", "\\\\?\\D:relative"])
def test_device_namespaces_refused_without_io(bundle, name):
    with pytest.raises(ValueError, match="namespace"):
        bundle.logical_path(name)


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended-path fixture")
def test_extended_unc_remains_outside_local_root_without_network_io(bundle, snapshot):
    path = bundle.logical_path("\\\\?\\UNC\\server\\share\\receipt.json")
    assert str(path) == "\\\\server\\share\\receipt.json"
    assert not path.is_relative_to(snapshot["canonical"])


@pytest.mark.skipif(os.name != "nt", reason="Win32 extended-path fixture")
def test_extended_credentials_path_is_still_refused(bundle, snapshot):
    secret = snapshot["write"](snapshot["model"].parent / "auth.json", b"fixture auth")
    with pytest.raises(ValueError, match="Credential/environment"):
        bundle.safe_file(bundle.io_path(secret), {"canonical":snapshot["canonical"]})


def reviewed_url_fixture(bundle, snapshot, *, prefix=b"", suffix=b""):
    """A machine syntax review binds one synthetic URL substring, never a token class."""
    raw = snapshot["write"](snapshot["canonical"] / "data/raw/reviewed.html",
        prefix + b'<a href="https://example.invalid/risk-' + b"public-article-" * 4 + b'2026">fixture</a>' + suffix)
    value = raw.read_bytes()
    match = bundle.SECRETS.search(value)
    assert match is not None
    row = dict(archive_path="canonical/data/raw/reviewed.html", file_sha256=bundle.sha(raw),
               byte_offset=match.start(), match_length=len(match.group()), match_sha256=hashlib.sha256(match.group()).hexdigest())
    review = dict(schema_version=1, classification="embedded_public_article_url_slug",
                  review_provenance="codex_syntax_review_not_human", findings=[row])
    review_path = snapshot["write"](snapshot["canonical"] / "data/machine_runs/private_reproducibility_review/review.json", review)
    snapshot["receipt"]["protected_sha256"][str(raw)] = bundle.sha(raw)
    snapshot["write"](snapshot["audit"], snapshot["receipt"])
    snapshot["args"]["audit_sha256"] = bundle.sha(snapshot["audit"])
    snapshot["args"]["benign_url_pattern_review"] = review_path
    return raw, review_path, review, row


def test_exact_machine_review_preserves_every_original_byte_and_manifest(bundle, snapshot):
    raw, review_path, review, row = reviewed_url_fixture(bundle, snapshot)
    before = raw.read_bytes()
    manifest = bundle.inventory(**snapshot["args"])
    recorded = manifest["credential_pattern_review"]
    assert recorded["review_provenance"] == "codex_syntax_review_not_human"
    assert recorded["reviewed_benign_findings"] == [row]
    assert recorded["review_file_sha256"] == bundle.sha(review_path)
    output = snapshot["canonical"] / "data/machine_runs/private_reproducibility/reviewed.zip"
    bundle.package(manifest, snapshot["canonical"], output)
    with zipfile.ZipFile(output) as archive:
        assert archive.read(row["archive_path"]) == before
        assert json.loads(archive.read(recorded["review_file"])) == review
        assert json.loads(archive.read("MANIFEST.json"))["credential_pattern_review"] == recorded
    assert raw.read_bytes() == before


@pytest.mark.parametrize("field,value", [
    ("archive_path", "canonical/models/model.onnx"),
    ("archive_path", "canonical/data/raw/../reviewed.html"),
    ("file_sha256", "0" * 64),
    ("byte_offset", 0),
    ("byte_offset", True),
    ("match_length", 129),
    ("match_length", True),
    ("match_sha256", "0" * 64),
])
def test_review_never_accepts_wrong_path_file_span_or_match_hash(bundle, snapshot, field, value):
    _, review_path, review, row = reviewed_url_fixture(bundle, snapshot)
    row[field] = value
    snapshot["write"](review_path, review)
    with pytest.raises(ValueError):
        bundle.inventory(**snapshot["args"])


@pytest.mark.parametrize("field,value", [("schema_version", True), ("classification", "safe_url"),
    ("review_provenance", "human_review"), ("findings", []), ("findings", {})])
def test_review_requires_explicit_machine_only_schema(bundle, snapshot, field, value):
    _, review_path, review, _ = reviewed_url_fixture(bundle, snapshot)
    review[field] = value
    snapshot["write"](review_path, review)
    with pytest.raises(ValueError, match="certificate"):
        bundle.inventory(**snapshot["args"])


def test_duplicate_review_finding_refused(bundle, snapshot):
    _, review_path, review, row = reviewed_url_fixture(bundle, snapshot)
    review["findings"].append(dict(row))
    snapshot["write"](review_path, review)
    with pytest.raises(ValueError, match="Duplicate"):
        bundle.inventory(**snapshot["args"])


def test_review_certificate_outside_exact_private_review_directory_refused(bundle, snapshot):
    _, _, review, _ = reviewed_url_fixture(bundle, snapshot)
    snapshot["args"]["benign_url_pattern_review"] = snapshot["write"](snapshot["canonical"] / "data/raw/review.json", review)
    with pytest.raises(ValueError, match="certificate path"):
        bundle.inventory(**snapshot["args"])


def test_review_is_not_a_global_url_or_embedded_token_exemption(bundle, snapshot):
    reviewed_url_fixture(bundle, snapshot)
    secret = b"sk-" + b"z" * 32
    snapshot["write"](snapshot["model"].parent / "unsafe.txt", b"x" * 262140 + secret)
    manifest = bundle.inventory(**snapshot["args"])
    output = snapshot["canonical"] / "data/machine_runs/private_reproducibility/unreviewed.zip"
    with pytest.raises(ValueError, match="credential") as failure:
        bundle.package(manifest, snapshot["canonical"], output)
    assert secret.decode() not in str(failure.value) and not output.exists()


def test_second_unreviewed_match_in_same_certified_source_is_refused(bundle, snapshot):
    reviewed_url_fixture(bundle, snapshot, suffix=b' <a href="https://example.invalid/?key=sk-' + b"z" * 32 + b'">bad</a>')
    manifest = bundle.inventory(**snapshot["args"])
    output = snapshot["canonical"] / "data/machine_runs/private_reproducibility/extra.zip"
    with pytest.raises(ValueError, match="credential"):
        bundle.package(manifest, snapshot["canonical"], output)
    assert not output.exists()


@pytest.mark.parametrize("remaining", [30, 80])
def test_review_absolute_offsets_and_carry_overlap_are_exact(bundle, snapshot, remaining):
    heading = b'<a href="https://example.invalid/ri'
    prefix = b"x" * (262144 - remaining - len(heading))
    _, _, _, row = reviewed_url_fixture(bundle, snapshot, prefix=prefix, suffix=b" " * 1000)
    assert row["byte_offset"] == 262144 - remaining
    manifest = bundle.inventory(**snapshot["args"])
    output = snapshot["canonical"] / f"data/machine_runs/private_reproducibility/carry-{remaining}.zip"
    bundle.package(manifest, snapshot["canonical"], output)
    assert output.exists()


def test_truncated_review_does_not_authorize_longer_regex_match(bundle, snapshot):
    raw, review_path, review, row = reviewed_url_fixture(bundle, snapshot)
    row["match_length"] -= 5
    partial = raw.read_bytes()[row["byte_offset"]:row["byte_offset"] + row["match_length"]]
    row["match_sha256"] = hashlib.sha256(partial).hexdigest()
    snapshot["write"](review_path, review)
    manifest = bundle.inventory(**snapshot["args"])
    output = snapshot["canonical"] / "data/machine_runs/private_reproducibility/truncated.zip"
    with pytest.raises(ValueError, match="credential"):
        bundle.package(manifest, snapshot["canonical"], output)
    assert not output.exists()


def test_mutated_review_certificate_is_refused_before_package(bundle, snapshot):
    _, review_path, _, _ = reviewed_url_fixture(bundle, snapshot)
    manifest = bundle.inventory(**snapshot["args"])
    review_path.write_bytes(review_path.read_bytes() + b"\n")
    output = snapshot["canonical"] / "data/machine_runs/private_reproducibility/changed-review.zip"
    with pytest.raises(ValueError, match="Inventory"):
        bundle.package(manifest, snapshot["canonical"], output)
    assert not output.exists()


def test_unused_review_finding_cannot_reach_final_publication(bundle, snapshot, monkeypatch):
    reviewed_url_fixture(bundle, snapshot)
    manifest = bundle.inventory(**snapshot["args"])
    original = bundle.SECRETS

    class NoObservedMatches:
        fullmatch = original.fullmatch
        search = original.search

        @staticmethod
        def finditer(_):
            return iter(())

    monkeypatch.setattr(bundle, "SECRETS", NoObservedMatches())
    output = snapshot["canonical"] / "data/machine_runs/private_reproducibility/unused.zip"
    with pytest.raises(ValueError, match="Unused"):
        bundle.package(manifest, snapshot["canonical"], output)
    assert not output.exists()
