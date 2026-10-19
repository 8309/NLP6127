"""Read-only source transformations: no invented table values or dates."""
import pytest

from oil_evidence import unified_corpus as uc
from oil_evidence.corpus import clean_page, split_passages
from oil_evidence.store import digest


def doc(source="DOE-SPR", **extra):
    return dict(document_id="fixture", source=source, title="Official oil report", **extra)


def test_observed_paragraph_boundaries_survive_pdf_line_wraps():
    text, removed = uc.clean_source_page(
        "Oil production rose in the reporting\nperiod.\n\nNASA leased office space."
    )
    assert text == "Oil production rose in the reporting period.\n\nNASA leased office space."
    assert removed == []


def test_inline_linked_report_title_is_not_a_running_heading():
    text, removed = uc.clean_source_page(
        "In the June\nShort-Term Energy Outlook\n, we forecast ethane exports will increase."
    )
    assert "Short-Term Energy Outlook" in text
    assert "\n\n" not in text
    assert removed == []


def test_metadata_removed_with_audit_but_units_and_dates_retained():
    text, removed = uc.clean_source_page(
        "September 2026\nOil prices\nBrent crude averaged $70 per barrel in August 2026.\n"
        "Source: official report\nU.S. crude oil production rose to 13.7 million barrels per day."
    )
    assert "$70 per barrel in August 2026" in text
    assert "13.7 million barrels per day" in text
    assert [r["reason"] for r in removed] == [
        "standalone_report_month", "standalone_section_heading", "source_credit_metadata"
    ]
    assert "September 2026" not in text
    assert "\n\n" in text


@pytest.mark.parametrize("footer", [
    "For questions about This Week in Petroleum, contact staff at 123-456-7890.",
    "Tags: petroleum, energy", "Principal contributors: analyst"
])
def test_footer_is_not_accepted_body(footer):
    text, removed = uc.clean_source_page("Oil supply increased.\n" + footer + "\nOther contact details.")
    assert text == "Oil supply increased."
    assert len(removed) == 2


@pytest.mark.parametrize("raw,source,expected", [
    ("AGRICULTURE\nCommodity Markets Outlook\nSoybean oil prices increased.", "WB-CMO", "explicit_non_oil_commodity_chapter"),
    ("Oil production rose.\nFIGURE 2.4 Supply chart 2018 2019 2020", "DOE-SPR", "mixed_figure_page_pending_layout_aware_extraction"),
    ("Oil imports increased.\nChart SF.2 Pipeline statistics", "EIA-STEO", "mixed_figure_page_pending_layout_aware_extraction"),
    ("The diversification index is of the form:\nWhere x means share.", "UK-ENERGY-TRENDS", "formula_page_pending_structured_extraction"),
    ("Asymmetric impacts of metal price shocks include oil comparators.", "WB-CMO", "explicit_non_oil_study_section"),
    ("U.S. energy market indicators 2025 2026 2027 Brent oil price 69 91 74", "EIA-STEO", "flattened_overview_indicator_table"),
    ("Description of price series Energy Crude oil. Oils and meals Coconut oil.", "WB-CMO", "mixed_price_series_dictionary_pending_structured_extraction"),
    ("Oil pr ices are forecast to rise. The m ain risks concern crops.", "WB-CMO", "fragmented_word_extraction_pending_layout_repair"),
    ("The price of crude oil increased but out put estimates are unavailable.", "EIA-STEO", "fragmented_word_extraction_pending_layout_repair"),
])
def test_ambiguous_layout_pages_are_held_not_reconstructed(raw, source, expected):
    assert uc.page_exclusion(doc(source), raw, 1, True) == expected


def test_recorded_old_exclusion_survives_new_ids():
    raw = "Crude oil production increased during the reporting period. " * 12
    old_text = split_passages(clean_page(raw), target=200, minimum=65)[0]
    old_id = "fixture-" + digest(" ".join(uc.words(old_text)))[:16]
    assert uc.page_exclusion(doc(excluded_passage_ids=[old_id]), raw, 1, False) == "recorded_legacy_passage_exclusion_requires_repair"


@pytest.mark.parametrize("broken", [
    "Seve n Sisters", "oil po ols", "des igned to stabilize oil prices",
    "T exas oil wells", "Septem ber oil output", "Liby a oil exports",
    "Nigeri a oil production", "rejo ined the oil producers' group",
])
def test_fractured_body_or_footnote_tokens_hold_any_pdf_source(broken):
    raw = "Oil production rose in 2025. " + broken
    assert uc.page_exclusion(doc("EIA-METHODS"), raw, 1, True) == "fragmented_word_extraction_pending_layout_repair"


def test_unbroken_names_are_not_layout_damage():
    raw = (
        "Texas oil output increased in September 2025. Nigeria and Libya "
        "exported crude oil. Prorationing was designed to stabilize oil prices. "
        "The company rejoined the oil producers' group."
    )
    # Other source-page gates can still reject this short fixture; the new
    # fractured-token detector must not mistake these normal spellings.
    assert uc.page_exclusion(doc("EIA-METHODS"), raw, 1, True) != "fragmented_word_extraction_pending_layout_repair"


def test_actual_html_table_subtree_is_excluded(tmp_path):
    raw = (b'<main><article><div class="field--name-body"><p>Crude oil production was suspended.</p>'
           b'<table><tr><td>Total Oil BOPD</td><td>12 34</td></tr></table>'
           b'<p>Operators will inspect the platforms.</p></div></article></main>')
    path = tmp_path / "bsee.html"
    path.write_bytes(raw)
    pages, extraction = uc.extract_existing(
        doc("BSEE-HURRICANE", raw_path="bsee.html", raw_sha256=digest(raw)), tmp_path
    )
    assert "Crude oil production was suspended." in pages[0]
    assert "Total Oil BOPD" not in pages[0]
    assert "12 34" not in pages[0]
    assert extraction["removed_html_tables"] == 1
    assert extraction["extraction_method"] == "recognized_html_body_tables_excluded_v2"


def test_new_html_extraction_fails_closed_without_recognized_body(tmp_path):
    raw = b"<main><table><tr><td>Oil 12</td></tr></table></main>"
    (tmp_path / "unknown.html").write_bytes(raw)
    with pytest.raises(ValueError, match="recognized article body"):
        uc.extract_existing(doc("BSEE-HURRICANE", raw_path="unknown.html", raw_sha256=digest(raw)), tmp_path)
