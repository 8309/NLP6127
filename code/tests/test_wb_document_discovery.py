import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def adapter():
    spec = importlib.util.spec_from_file_location('wb_documents', Path(__file__).parents[1] / 'scripts/discover_wb_documents.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_main_document_actual_anchor_only_not_related_or_guessed(adapter):
    html = '<h4>MAIN DOCUMENT</h4><ul><li><a href="original/CMO-April-2025.pdf">CMO-April-2025.pdf</a></li></ul><h4>RELATED</h4><a href="original/CMO-April-2025-Forecasts.pdf">Forecasts</a><a href="https://other.invalid/CMO-April-2025.pdf">Other source</a>'
    origin = adapter.OBSERVED_LANDINGS[0][0]
    row = adapter.document_sources(html, origin, 2025, 4)[0]
    # urljoin is the only resolution: no '/original/' path is manufactured.
    assert row['url'] == 'https://thedocs.worldbank.org/en/doc/1b388949805c9a0ae3736bdacb32ea94-0050012025/original/CMO-April-2025.pdf'
    assert row['available_by'] == '2025-04-30' and row['published_date'] is None
    assert row['date_source'] == adapter.ARCHIVE and row['requires_pdf'] is True
    assert row['redistribution'] == 'review_required'


def test_no_main_document_or_wrong_issue_fail_closed(adapter):
    origin = adapter.OBSERVED_LANDINGS[0][0]
    assert adapter.document_sources('<h4>RELATED</h4><a href="original/CMO-April-2025.pdf">PDF</a>', origin, 2025, 4) == []
    assert adapter.document_sources('<h4>MAIN DOCUMENT</h4><a href="original/CMO-October-2025.pdf">PDF</a>', origin, 2025, 4) == []
    assert adapter.document_sources('<h4>MAIN DOCUMENT</h4><a href="https://bit.ly/report">PDF</a>', origin, 2025, 4) == []


def test_official_docs1_identifier_pdf_only_with_exact_main_document_label(adapter):
    html = '<h4>MAIN DOCUMENT</h4><a href="https://documents1.worldbank.org/curated/en/123/pdf/IDU-observed.pdf">CMO-April-2025.pdf</a>'
    rows = adapter.document_sources(html, adapter.OBSERVED_LANDINGS[0][0], 2025, 4)
    assert len(rows) == 1 and rows[0]['url'].endswith('IDU-observed.pdf')
    assert adapter.document_sources(html.replace('CMO-April-2025.pdf', 'CMO-October-2025.pdf'), adapter.OBSERVED_LANDINGS[0][0], 2025, 4) == []


def test_no_official_domain_credentials_or_out_of_window(adapter):
    html = '<h4>MAIN DOCUMENT</h4><a href="https://thedocs.worldbank.org/original/CMO-April-2025.pdf">PDF</a>'
    assert adapter.document_sources(html, 'https://not-worldbank.invalid/', 2025, 4) == []
    html = '<h4>MAIN DOCUMENT</h4><a href="https://user:password@thedocs.worldbank.org/original/CMO-April-2025.pdf">PDF</a>'
    assert adapter.document_sources(html, adapter.OBSERVED_LANDINGS[0][0], 2025, 4) == []
    assert adapter.document_sources(html, adapter.OBSERVED_LANDINGS[0][0], 2017, 4) == []


def test_discover_fetches_exactly_two_observed_landings_keeps_failure_no_retry(adapter, monkeypatch, tmp_path):
    import oil_evidence.store as store
    monkeypatch.setattr(store, 'DATA', tmp_path)
    class Fetch:
        def __init__(self):
            self.urls = []
        def get(self, url):
            self.urls.append(url)
            if len(self.urls) == 2:
                raise RuntimeError('403 Forbidden')
            return b'<h4>MAIN DOCUMENT</h4><a href="https://thedocs.worldbank.org/en/doc/a/original/CMO-April-2025.pdf">PDF</a>', url, 'text/html'
    fetch = Fetch()
    rows = adapter.discover(fetch)
    assert len(rows) == 1 and fetch.urls == [item[0] for item in adapter.OBSERVED_LANDINGS]
    assert len(list((tmp_path/'expansion_runs/official-2018-2026-v2/discovery/wb-documents').glob('*.html'))) == 1
    assert not (tmp_path/'sources.jsonl').exists()
    assert not (tmp_path/'documents.jsonl').exists()
