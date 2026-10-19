import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/discover_eia_buffer.py"
spec = importlib.util.spec_from_file_location("discover_eia_buffer", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def article(extra="", day="March 7, 2024", title="U.S. gasoline exports increased in 2023", body=None):
    prose = body or " ".join([
        "U.S. gasoline exports increased in 2023 as refinery production expanded and petroleum demand recovered.",
        "The crude oil market supplied refineries with feedstock, and the resulting motor gasoline was shipped abroad.",
        "The report measures barrels per day rather than total barrels and separates petroleum products from crude oil.",
        "In 2023, the United States produced petroleum products for domestic consumers and international buyers.",
        "The forecast considers refinery maintenance, international trade, fuel demand, inventory changes, and available transport capacity.",
        "These observations describe the reported statistical period and do not establish future realized production outcomes.",
        "Refineries converted imported and domestic crude oil into products distributed through pipelines and terminals in 2023.",
    ])
    return f"<nav>January 1, 2026</nav><div class='tie-article'><span class='date'>{day}</span><h1>{title}</h1><p>{prose}</p>{extra}</div>".encode()


@pytest.mark.parametrize("url", [
    "http://www.eia.gov/todayinenergy/detail.php?id=9",
    "https://www.eia.gov.evil.test/todayinenergy/detail.php?id=9",
    "https://www.eia.gov@evil.test/todayinenergy/detail.php?id=9",
    "https://user@www.eia.gov/todayinenergy/detail.php?id=9",
    "https://www.eia.gov:444/todayinenergy/detail.php?id=9",
    "https://www.eia.gov/todayinenergy/detail.php?id=9&id=10",
    "https://www.eia.gov/todayinenergy/detail.php?id=9&x=1",
    "https://www.eia.gov/todayinenergy/detail.php?id=x",
    "https://www.eia.gov/todayinenergy/detail.php?id=9#other",
    "https://www.eia.gov/other/detail.php?id=9",
])
def test_route_rejects_untrusted_or_ambiguous(url):
    with pytest.raises(ValueError):
        mod.admitted_eia_route(url)


def test_exact_official_route():
    assert mod.admitted_eia_route("https://www.eia.gov/todayinenergy/detail.php?id=9") == "https://www.eia.gov/todayinenergy/detail.php?id=9"


def test_linked_years_are_observed_not_guessed():
    raw = b"<a href='archive.php?my=2024'>2024</a><a href='archive.php?my=2018'>2018</a><a href='archive.php?my=2017'>2017</a><a href='https://evil.test/archive.php?my=2025'>2025</a>"
    assert mod.linked_year_archives(raw, mod.ARCHIVE) == [
        "https://www.eia.gov/todayinenergy/archive.php?my=2018",
        "https://www.eia.gov/todayinenergy/archive.php?my=2024"]


def test_archive_broader_titles_still_require_item_date_and_scope():
    raw = b"""<ul>
    <li>March 7, 2024 <a href='detail.php?id=1'>U.S. gasoline demand increased</a></li>
    <li>March 8, 2024 <a href='detail.php?id=2'>Solar panel generation rose</a></li>
    <li>March 9, 2024 <a href='detail.php?id=3'>Refinery crude oil runs increased</a></li>
    <li><a href='detail.php?id=4'>Diesel exports rose</a></li>
    <li>October 6, 2026 <a href='detail.php?id=5'>Petroleum outlook</a></li>
    </ul>"""
    known = {"https://www.eia.gov/todayinenergy/detail.php?id=3"}
    rows, exclusions, count = mod.parse_archive(raw, mod.ARCHIVE, known)
    assert count == 5 and len(rows) == 1
    assert rows[0]["source"] == "EIA-OIL-NEWS" and rows[0]["available_by"] == "2024-03-07"
    assert rows[0]["body_oil_verified"] is False
    assert exclusions == {"not_oil_fuel_transport_title": 1, "already_registered": 1,
                          "missing_article_local_archive_day": 1, "outside_window": 1}


def test_article_local_day_and_body():
    parsed = mod.parse_eia_body(article())
    assert parsed["published_date"] == "2024-03-07"
    assert parsed["title"] == "U.S. gasoline exports increased in 2023"
    assert parsed["body_oil_verified"] is True
    assert "January 1, 2026" not in parsed["text"]


def test_figure_table_caption_and_footer_not_prose():
    extra = "<table><tr><td>BADTABLE</td></tr></table><figure><p>BADFIGURE</p></figure><div class='source'><p>BADCAPTION</p></div><footer><p>BADFOOTER</p></footer><p>Tags: BADTAGS</p>"
    parsed = mod.parse_eia_body(article(extra))
    assert "BAD" not in parsed["text"]


def test_unclosed_image_wrapper_retains_paragraph_boundaries_once():
    raw = article().replace(b"<p>U.S.", b"<p data-type='inbrief'><img src='chart.svg'><p>U.S.")
    parsed = mod.parse_eia_body(raw)
    assert len(parsed["blocks"]) == 2
    assert parsed["text"].count("U.S. gasoline exports increased") == 2  # title plus one body occurrence


def test_list_paragraphs_appear_once():
    parsed = mod.parse_eia_body(article("<ul><li><p>Oil exports increased in 2024.</p></li></ul>"))
    assert parsed["text"].count("Oil exports increased in 2024.") == 1


@pytest.mark.parametrize("change", [
    lambda raw: raw.replace(b"class='tie-article'", b"class='unknown'"),
    lambda raw: raw.replace(b"class='date'", b"class='unknown'"),
    lambda raw: raw.replace(b"<h1>", b"<h2>").replace(b"</h1>", b"</h2>"),
    lambda raw: raw.replace(b"March 7, 2024", b"Unknown date"),
])
def test_missing_native_article_metadata_fails_closed(change):
    with pytest.raises(ValueError):
        mod.parse_eia_body(change(article()))


def test_subsequent_correction_fails_closed():
    with pytest.raises(ValueError, match="later revision"):
        mod.parse_eia_body(article("<p>Updated March 12, 2024 to correct the data.</p>"))


def test_same_day_notice_is_not_later_revision():
    assert mod.parse_eia_body(article("<p>Updated March 7, 2024 to correct the data.</p>"))["published_date"] == "2024-03-07"


@pytest.mark.parametrize("notice", [
    "Correction: This article was revised.",
    "Editor's note: The figures in this article have been corrected.",
    "This article has been updated to correct the figures.",
    "Updated June 12 to clarify EU limits on the share of biofuels.",
])
def test_explicit_undated_revision_notice_fails_closed(notice):
    with pytest.raises(ValueError, match="undated revision notice"):
        mod.parse_eia_body(article(f"<p>{notice}</p>"))


@pytest.mark.parametrize("notice", [
    "Updated: 2024-03-12 to correct the data.",
    "Editor’s note: The article was corrected on 2024-03-12.",
    "Correction: First issued March 7, 2024; corrected on 2024-03-12.",
])
def test_iso_or_multiple_day_later_revision_fails_closed(notice):
    with pytest.raises(ValueError, match="later revision"):
        mod.parse_eia_body(article(f"<p>{notice}</p>"))


@pytest.mark.parametrize("prose", [
    "EIA revised estimates on March 12, 2024 for the statistical series discussed in this report.",
    "Revised estimates for refinery production were released on March 12, 2024.",
])
def test_ordinary_revised_estimates_are_not_article_revision_notices(prose):
    parsed = mod.parse_eia_body(article(f"<p>{prose}</p>"))
    assert parsed["published_date"] == "2024-03-07"
    assert prose in parsed["text"]


def test_same_day_iso_notice_preserved_and_caption_notice_not_used():
    extra = ("<p>Updated: 2024-03-07 to correct the data.</p>"
             "<figure><p>Updated: 2024-03-12 third-party chart.</p></figure>"
             "<div class='source'><p>Correction: External image.</p></div>")
    parsed = mod.parse_eia_body(article(extra))
    assert parsed["published_date"] == "2024-03-07"
    assert "Updated: 2024-03-07" in parsed["text"]
    assert "third-party chart" not in parsed["text"]
    assert "External image" not in parsed["text"]


def test_non_oil_transport_title_can_fail_body_scope_without_becoming_gold():
    body = " ".join(["Passenger vehicles use electrical power from rechargeable batteries in cities and towns."] * 12)
    parsed = mod.parse_eia_body(article(title="Passenger vehicles increased in 2023", body=body))
    assert parsed["body_oil_verified"] is False


def test_pilot_selection_is_deterministic_date_spread():
    rows = [dict(url=f"u{i}", available_by=f"2024-01-{i + 1:02d}") for i in range(20)]
    assert mod.select_pilots(rows, 8) == mod.select_pilots(list(reversed(rows)), 8)
    picks = mod.select_pilots(rows, 8)
    assert len(picks) == 8 and picks[0]["url"] == "u0" and picks[-1]["url"] == "u19"
