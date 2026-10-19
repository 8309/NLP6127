import importlib.util
from pathlib import Path
import pytest
from oil_evidence.store import read_jsonl, write_jsonl, digest


@pytest.fixture
def expansion():
    spec = importlib.util.spec_from_file_location('expansion', Path(__file__).parents[1]/'scripts/expand_corpus.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_twip_discovery_keeps_oil_background_not_only_keyword_titles(expansion):
    html = '<a href="/petroleum/weekly/archive/2021/210210/includes/analysis_print.php">Gasoline demand</a><a href="/petroleum/weekly/archive/2017/170210/includes/analysis_print.php">Oil</a>'
    rows = expansion.twip_sources(html,'https://www.eia.gov/petroleum/weekly/')
    assert len(rows)==1 and rows[0]['available_by']=='2021-02-10'


def test_country_dates_are_release_dates_not_event_dates(expansion):
    html = 'Last Updated: October 10, 2024 <a href="content/analysis/countries_long/Iran/pdf/current.pdf">PDF</a><a href="content/analysis/countries_long/Iran/archive/old.pdf">2021</a>'
    rows = expansion.country_sources(html,'https://www.eia.gov/international/content/analysis/countries_long/Iran/')
    assert rows[0]['available_by']=='2024-10-10'
    assert '/international/content/analysis/' in rows[0]['url'] and '/Iran/content/' not in rows[0]['url']
    assert rows[1]['available_by']=='2021-12-31' and rows[1]['published_date'] is None
    assert rows[1]['date_precision']=='year_upper_bound'


def test_worldbank_opaque_links_and_out_of_window_rejected(expansion):
    html = '<table><tr><td>2024 - April</td><td><a href="https://bit.ly/private">PDF</a></td></tr><tr><td>2020 - Oct</td><td><a href="https://thedocs.worldbank.org/report.pdf">PDF</a></td></tr><tr><td>2017 - Oct</td><td><a href="https://thedocs.worldbank.org/old.pdf">PDF</a></td></tr></table>'
    rows = expansion.worldbank_sources(html,'https://www.worldbank.org/archive')
    assert len(rows)==1 and rows[0]['available_by']=='2020-10-31'


def test_incremental_build_preserves_exact_baseline(tmp_path,monkeypatch):
    import oil_evidence.corpus as corpus
    monkeypatch.setattr(corpus,'DATA',tmp_path)
    # This is a historical incremental-builder test, not a request to mix old
    # chunks into the uniform source-page pipeline selected by live config.
    monkeypatch.setattr(corpus, 'config', lambda: {'corpus': {
        'segmentation_version': 'legacy-incremental-fixture',
        'chunk_words': 200, 'min_words': 65,
        'start_date': '2018-01-01', 'end_date': '2026-10-05',
    }})
    old=dict(passage_id='original',document_id='old',text='Original crude oil evidence.',word_count=4,source='EIA',available_by='2024-01-01',date_precision='day')
    text='Crude oil production was stable in the country during the annual reporting period. '*12
    raw=('<article>'+text+'</article>').encode()
    (tmp_path/'raw').mkdir()
    (tmp_path/'raw/new.html').write_bytes(raw)
    write_jsonl(tmp_path/'documents.jsonl',[dict(document_id='old',raw_path='missing-old.pdf',raw_sha256='unused'),dict(document_id='new',raw_path='raw/new.html',raw_sha256=digest(raw),source='EIA',available_by='2024-01-01',title='New crude oil report')])
    staged=tmp_path/'stage'
    report=corpus.build_corpus(output_dir=staged,baseline=[old])
    assert read_jsonl(staged/'passages.jsonl')[0]==old
    assert report['passages']>1 and report['retained_baseline_passages']==1 and not report['extraction_errors']
    assert not (tmp_path/'passages.jsonl').exists()


def test_country_bibliographies_are_not_evidence():
    from oil_evidence.corpus import is_reference_page
    text='1 Reuters, Oil exports, https://example.org/1\n2 EIA, Accessed March 2024\n3 Middle East Economic Survey, Production\n'
    assert is_reference_page(text)
    assert is_reference_page('Endnotes\n1 Oil data source')
    assert not is_reference_page('Crude oil production increased by 3 million barrels per day.\nThe report discusses three scenarios for production growth.')


def test_country_registry_only_selects_safe_official_long_analysis(expansion):
    rows = expansion.country_pages([dict(country='Saudi_Arabia', longAnalysis=True),
        dict(country='Belgium', longAnalysis=False), dict(country='../secret', longAnalysis=True)])
    assert len(rows)==1
    assert rows[0][0]=='https://www.eia.gov/international/content/analysis/countries_long/Saudi_Arabia/'


def test_oil_scope_not_just_corporation_or_vegetable_oil():
    from oil_evidence.corpus import oil_topic_supported,has_table_heading
    assert not oil_topic_supported('The LNG plant is owned by China National Petroleum Corporation and China National Offshore Oil Corporation.')
    assert not oil_topic_supported('Most biodiesel is produced from soybean oil.')
    assert oil_topic_supported('Gas flaring at crude oil production facilities is a supply risk.')
    assert has_table_heading('Overview\nTable 3. Refinery capacity\nName Capacity')
    assert not has_table_heading('As discussed in Table 3, crude oil production increased.')


def test_unlabelled_numeric_chart_ticks_are_not_prose():
    from oil_evidence.corpus import clean_page
    assert clean_page('Oil production increased.\n0.0\n0.2\n0.4\nFigure 1.')=='Oil production increased. Figure 1.'
