"""Synthetic rights-ledger checks, not legal opinions or permission evidence."""
from copy import deepcopy
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import preflight_data_rights as module


def row(source="EIA", url="https://www.eia.gov/outlooks/steo/archives/sep26.pdf", **changes):
    result = dict(passage_id="fixture-p", document_id="fixture-d", source=source, url=url,
        text="Synthetic secret-like corpus text never enters the registry.", title="Synthetic title",
        redistribution="eia_attribution", available_by="2024-01-01")
    result.update(changes)
    return result


def test_candidate_is_not_clearance_and_original_flags_and_text_are_unchanged():
    original = [row()]
    before = deepcopy(original)
    registry, documents, summary = module.inventory(original)
    assert original == before
    assert summary["conditional_candidate_passages"] == 1 and summary["text_export_cleared_passages"] == 0
    assert registry[0]["original_redistribution"] == "eia_attribution" and not registry[0]["export_text_cleared"]
    assert "text" not in registry[0] and "text" not in documents[0]
    assert original[0]["text"] not in module.protocol.canonical_bytes(registry).decode()


@pytest.mark.parametrize("source,url,expected", [
    ("DALLAS-OIL","https://www.dallasfed.org/research/surveys/des/2019/1901","held"),
    ("BOEM-EIS","https://www.boem.gov/example.pdf","held"),
    ("TREASURY-OIL","https://home.treasury.gov/news/press-releases/example","held"),
    ("OFAC","https://ofac.treasury.gov/recent-actions/20260827","held"),
    ("WB-CMO",module.WB_OCTOBER,"candidate_conditional"),
    ("WB-CMO","https://thedocs.worldbank.org/unreviewed-report.pdf","held"),
    ("EIA","https://www.eia.gov.attacker.example/article","held"),
    ("UNREVIEWED","https://www.eia.gov/article","held"),
])
def test_hosting_domain_or_another_report_licence_does_not_establish_rights(source,url,expected):
    assert module.decision(row(source,url))[1] == expected


def test_duplicate_ids_and_conflicting_document_urls_rejected():
    with pytest.raises(ValueError, match="duplicated"):
        module.inventory([row(),row()])
    with pytest.raises(ValueError, match="conflicting"):
        module.inventory([row(), row(passage_id="fixture-other", url="https://www.eia.gov/different")])


def test_registry_zip_contains_only_explicit_metadata_allowlist_and_preserves_original(tmp_path):
    corpus = tmp_path / "data/frozen/fixture/corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    raw = module.protocol.canonical_bytes(row())+b"\n"
    corpus.write_bytes(raw)
    output, receipt = module.run(tmp_path, corpus, expected_sha256=module.protocol.sha256(raw))
    assert corpus.read_bytes() == raw and receipt["original_corpus_unchanged"]
    with zipfile.ZipFile(output / "source-registry-only.zip") as archive:
        assert set(archive.namelist()) == set(receipt["registry_zip_contents"])
        assert len(archive.namelist()) == 3
        assert all(row()["text"].encode() not in archive.read(name) for name in archive.namelist())
    assert receipt["raw_text_exported"] is False and receipt["legal_certification"] is False


def test_wrong_original_hash_fails_before_output_creation(tmp_path):
    corpus = tmp_path / "data/frozen/fixture/corpus.jsonl"
    corpus.parent.mkdir(parents=True)
    corpus.write_bytes(module.protocol.canonical_bytes(row())+b"\n")
    with pytest.raises(ValueError, match="hash differs"):
        module.run(tmp_path, corpus, expected_sha256="0"*64)
    assert not (tmp_path / "data/machine_runs/data-rights").exists()


def test_policy_quotes_are_short_and_inspection_date_is_not_falsely_refreshed():
    assert all(len(entry["short_quote"].split()) <= 5 for entry in module.EVIDENCE.values())
    assert module.OBSERVED_ON == "2026-10-13"
    assert module.EVIDENCE["ogl"]["observation"] == "web_fetch_403_no_bypass"
