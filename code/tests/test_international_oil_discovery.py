import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def adapters():
    spec = importlib.util.spec_from_file_location("international_oil", Path(__file__).parents[1] / "scripts/discover_international_oil.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_req_archive_uses_observed_landing_links_only(adapters):
    html = '<main><a href="/node/97117">Resources and energy quarterly: September 2026</a><a href="/publications/resources-and-energy-quarterly-june-2025">Resources and energy quarterly: June 2025</a><a href="https://thirdparty.invalid/full">Resources and energy quarterly: March 2025</a><a href="/publications/resources-and-energy-quarterly-december-2026">Resources and energy quarterly: December 2026</a></main>'
    rows = adapters.req_archive_pages(html)
    assert len(rows) == 2
    assert rows[0]["url"] == "https://www.industry.gov.au/node/97117"
    assert rows[1]["date_precision"] == "month_upper_bound"


def test_req_pdf_exact_publish_date_dedup_size_and_edition_guards(adapters):
    pdf = '/sites/default/files/2025-06/resources-and-energy-quarterly-june-2025.pdf'
    html = f'<main><h1>Resources and energy quarterly: June 2025</h1><div class="field--name-field-date"><div class="field__item">30 June 2025</div></div><a href="{pdf}"><span class="file-size">8.33 MB</span></a><a href="{pdf}">Duplicate</a><a href="/sites/default/files/2025-03/resources-and-energy-quarterly-march-2025.pdf">Other issue</a><a href="https://other.invalid/{pdf}">Other host</a></main>'
    rows = adapters.req_publication_sources(html, 'https://www.industry.gov.au/publications/resources-and-energy-quarterly-june-2025')
    assert len(rows) == 1 and rows[0]['published_date'] == '2025-06-30'
    assert rows[0]['redistribution'] == 'cc_by_4_attribution_with_exclusions'
    assert rows[0]['requires_pdf'] is True
    assert adapters.req_publication_sources(html.replace('8.33 MB', '25.01 MB').replace(f'<a href="{pdf}">Duplicate</a>', ''), 'https://www.industry.gov.au/publications/resources-and-energy-quarterly-june-2025') == []


def test_req_missing_exact_date_is_conservative_not_invented(adapters):
    html = '<h1>Resources and energy quarterly: September 2026</h1><a href="/sites/default/files/2026-09/resources-and-energy-quarterly-september-2026.pdf">PDF</a>'
    row = adapters.req_publication_sources(html, 'https://www.industry.gov.au/publications/resources-and-energy-quarterly-september-2026')[0]
    assert row['available_by'] == '2026-09-30' and row['published_date'] is None


def test_uk_archive_ignores_press_release_and_out_of_window(adapters):
    html = '<li><a href="/government/statistics/energy-trends-september-2025">Energy Trends: September 2025</a><time datetime="2025-09-30">30 September 2025</time></li><a href="/government/news/oil-release">UK oil statistics press release</a><a href="/government/statistics/energy-trends-december-2026">Energy Trends: December 2026</a>'
    rows = adapters.uk_energy_trends_pages(html)
    assert len(rows) == 1 and rows[0]['available_by'] == '2025-09-30'


def test_uk_revision_date_conservative_and_future_rejected(adapters):
    html = '<main><h1>Energy Trends: September 2025</h1><dt class="gem-c-metadata__term">Published:</dt><dd>30 September 2025</dd><section class="gem-c-attachment"><h3 class="gem-c-attachment__title"><a href="https://assets.publishing.service.gov.uk/media/abc/Energy_Trends_September_2025.pdf">Report</a></h3>PDF, 1.5 MB</section></main><div class="gem-c-published-dates">Last updated 2 February 2026</div>'
    rows = adapters.uk_energy_trends_sources(html, 'https://www.gov.uk/government/statistics/energy-trends-september-2025')
    assert rows[0]['available_by'] == '2026-02-02'
    assert rows[0]['original_publication_date'] == '2025-09-30'
    assert rows[0]['historical_attachment_version'] == 'not_independently_verified'
    assert adapters.uk_energy_trends_sources(html.replace('2 February 2026', '10 October 2026'), 'https://www.gov.uk/government/statistics/energy-trends-september-2025') == []


def test_worldbank_thematic_candidates_oil_only_direct_links(adapters):
    html = '<table><tr><td>2018 - April</td><td><a href="https://thedocs.worldbank.org/en/doc/a/original/focus.pdf">Oil exporters: Policies and challenges</a><a href="https://bit.ly/CMO">Oil supply shock</a><a href="https://thedocs.worldbank.org/metals.pdf">Metal price shocks</a></td></tr><tr><td>2017 - Oct</td><td><a href="https://thedocs.worldbank.org/old.pdf">Oil policy</a></td></tr></table>'
    rows = adapters.worldbank_special_oil_sources(html)
    assert len(rows) == 1 and rows[0]['available_by'] == '2018-04-30'


def test_orchestrator_discover_uses_supplied_fetcher_without_manifest_writes(adapters):
    archive = '<main><a href="/publications/resources-and-energy-quarterly-june-2025">Resources and energy quarterly: June 2025</a></main>'
    edition = '<h1>Resources and energy quarterly: June 2025</h1><a href="/sites/default/files/2025-06/resources-and-energy-quarterly-june-2025.pdf">Full report</a>'
    class FixtureFetcher:
        def __init__(self):
            self.visited = []
        def get(self, url):
            self.visited.append(url)
            return (archive if url == adapters.REQ_ARCHIVE else edition).encode(), url, 'text/html'
    fetch = FixtureFetcher()
    rows = adapters.discover(fetch, families=['AU-REQ'])
    assert len(rows) == 1 and len(fetch.visited) == 2
    assert rows[0]['archive_url'] == adapters.REQ_ARCHIVE


def test_dukes_archive_date_not_statistical_reporting_year(adapters):
    html = '<li><a href="/government/statistics/digest-of-uk-energy-statistics-dukes-2026">Digest of UK Energy Statistics (DUKES) 2026</a><time datetime="2026-07-30">30 July 2026</time></li><a href="/government/statistics/petroleum-dukes">Digest of UK Energy Statistics (DUKES): petroleum</a>'
    rows = adapters.uk_dukes_pages(html)
    assert len(rows) == 1 and rows[0]['available_by'] == '2026-07-30'
    html = '<main><h1>Digest of UK Energy Statistics (DUKES) 2025</h1><dt class="gem-c-metadata__term">Published:</dt><dd>31 July 2025</dd><section class="gem-c-attachment"><h3 class="gem-c-attachment__title"><a href="https://assets.publishing.service.gov.uk/media/abc/DUKES_2025.pdf">Report</a></h3>PDF, 10 MB</section></main>'
    row = adapters.uk_dukes_sources(html, 'https://www.gov.uk/government/statistics/digest-of-uk-energy-statistics-dukes-2025')[0]
    assert row['source'] == 'UK-DUKES' and row['available_by'] == '2025-07-31'
