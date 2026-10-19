import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def scope():
    spec = importlib.util.spec_from_file_location("oil_pdf_scope", Path(__file__).parents[1] / "scripts/oil_pdf_scope.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.oil_page_numbers


def test_non_special_series_unrestricted(scope):
    assert scope(['Oil prices rose'], 'UK-ENERGY-TRENDS') is None
    assert scope(['Oil prices rose'], 'EIA') is None


def test_au_actual_duplicated_extraction_heading_and_toc(scope):
    pages = ['Resources and Energy Quarterly | June 2025 iii\nContents\nGas 49\nOil 58\nUranium 65',
             'Resources and Energy Quarterly | June 2025 57\nTable 6.1: Gas outlook\nJCCC oil price',
             'Resources and Energy Quarterly | June 2025 58\n7. OilOil\nAustralia’s oil sector',
             'Resources and Energy Quarterly | June 2025 59\nOil trade map',
             'Resources and Energy Quarterly | June 2025 60\n7.1 Summary\nOil production increased',
             'Resources and Energy Quarterly | June 2025 64\nTable 7.1: Oil Outlook',
             'Resources and Energy Quarterly | June 2025 65\n8. Uranium Uranium\nAustralia’s uranium sector']
    assert scope(pages, 'AU-REQ') == {3, 4, 5, 6}


def test_au_only_toc_keywords_or_missing_boundary_fail_closed(scope):
    assert scope(['Contents\nOil 58\nUranium 65', 'Iron ore prices depend on oil costs'], 'AU-REQ') == set()
    assert scope(['7. OilOil\nAustralia’s oil sector', 'Oil production prose'], 'AU-REQ') == set()
    assert scope(['Gas outlook\nOil\nprice effects', '8. Uranium Uranium'], 'AU-REQ') == set()


def test_au_march_2025_actual_numbered_cover_heading_at_ninth_line(scope):
    # Text-layer fixtures from raw 68ae0f7846b7e6dfb677.pdf, physical 3/65/72.
    pages = [''] * 72
    pages[2] = ('Resources and Energy Quarterly | March 2025 3\nContents\n'
                'Gas 54\nOil 65\nUranium 72\nGold 78')
    pages[64] = ('Resources and Energy Quarterly | March 2025 65\n'
                 '1.\n2.\n3.\n4.\n5.\n6.\n7. Oil')
    pages[71] = ('Resources and Energy Quarterly | March 2025 72\n'
                 '1.\n2.\n3.\n4.\n5.\n6.\n7.\n8. Uranium')
    assert scope(pages, 'AU-REQ') == set(range(65, 72))


def test_au_raster_chapter_covers_still_fail_closed(scope):
    # December 2022's TOC has chapter names, but the covers' text layer is
    # only page numbers. A TOC alone must not authorize whole-chapter prose.
    pages = ['Contents\nOil 82\nUranium 93', '82', '8.1 Summary\nOil prose', '93']
    assert scope(pages, 'AU-REQ') == set()


def test_au_toc_boundary_disagreement_fail_closed(scope):
    pages = ['Contents\nOil 58\nUranium 65', 'Resources and Energy Quarterly | June 2025 58\n7. OilOil',
             'Resources and Energy Quarterly | June 2025 61\n8. Uranium Uranium']
    assert scope(pages, 'AU-REQ') == set()


def test_uk_chapter_start_end_and_gas_heading_at_text_layer_end(scope):
    pages = ['Contents\nChapter 3: Petroleum\nChapter 4: Gas',
             'Chapter 2: Solid Fuels and Derived Gases\nCoal uses oil transport',
             'Chapter 3: Oil and Oil Products\nKey headlines\nCrude production fell',
             'Oil production and stocks', 'Petroleum flow chart 2024',
             'Chart 4.1 Supply and demand for natural gas\nKey headlines\nGas supply\nChapter 4: Natural Gas\nEdward John Beazleigh',
             'Chapter 5: Electricity\nOil generation declined']
    assert scope(pages, 'UK-DUKES') == {3, 4, 5}


def test_uk_references_are_not_headings_and_unknown_boundary_fail_closed(scope):
    assert scope(['See Chapter 3: Petroleum for oil statistics', 'Chapter 4: Natural Gas'], 'UK-DUKES') == set()
    assert scope(['Chapter 3: Petroleum\nOil text', 'Chapter 5: Electricity'], 'UK-DUKES') == set()
    assert scope(['Chapter 3: Petroleum\nOil text'], 'UK-DUKES') == set()
    assert scope(['Data tables in this release\nChapter 3: Petroleum\nChapter 4: Natural Gas'], 'UK-DUKES') == set()


def test_ambiguous_duplicate_oil_chapters_fail_closed(scope):
    assert scope(['Chapter 3: Petroleum', 'Chapter 3: Petroleum', 'Chapter 4: Natural Gas'], 'UK-DUKES') == set()
    assert scope(['7. OilOil', '7. OilOil', '8. Uranium Uranium'], 'AU-REQ') == set()


def test_real_downloaded_samples_when_available(scope):
    from pypdf import PdfReader
    root = Path(__file__).parents[1] / 'data/expansion_runs/official-2018-2026-v2/discovery/oil-scope-samples'
    au = list(root.glob('AU-REQ-ae3a19fe54fb4f86.pdf'))
    uk = list(root.glob('UK-DUKES-ca4346ddfbe91538.pdf'))
    if not au or not uk:
        pytest.skip('Read-only official samples are not bundled as test fixtures')
    assert scope([page.extract_text() or '' for page in PdfReader(au[0]).pages], 'AU-REQ') == set(range(58, 65))
    assert scope([page.extract_text() or '' for page in PdfReader(uk[0]).pages], 'UK-DUKES') == set(range(13, 21))
