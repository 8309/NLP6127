import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("discover_dallas_buffer", Path(__file__).parents[1] / "scripts/discover_dallas_buffer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def survey(day="First Quarter | March 27, 2019", title="Oil and Gas Activity Grows Modestly as Oil Price Jumps", extra="", header=True):
    prose = " ".join([
        "Energy sector activity grew modestly in first quarter 2019 according to oil and gas executives responding to the Dallas Fed Energy Survey.",
        "The oil production index decreased from 29.1 in the fourth quarter to 21.1 in the first quarter, indicating slower growth reported by firms.",
        "Executives expect West Texas Intermediate oil prices at year end 2019 to average $60.19 per barrel.",
        "Responses ranged from $43 to $76 per barrel during the quarter, and these expectations are not realized future petroleum prices.",
        "The survey measures respondents' reported activity and does not establish total physical crude oil production in barrels per day.",
    ])
    date_header = f"<h5 class='survey-date'>{day}</h5>" if header else ""
    return f"""<html><head><meta property='og:title' content='{title}'></head>
    <nav>January 1, 2026</nav><div id='content'><div class='dal-main-content'>
    <h1 class='dal-headline'>Dallas Fed Energy Survey</h1>
    <div id='des-report'>{date_header}<h2>{title}</h2><p>{prose}</p>{extra}
    <p><strong>Next release:</strong> June 26, 2019</p></div>
    <div id='des-questions'><h5 id='des-reportdate'>{day}</h5><h2>Special Questions</h2>
    <h4>What WTI oil price does your firm need to drill a new well?</h4><p>Firms reported $50 per barrel on average in 2019.</p></div>
    <div id='des-comments'><h5>{day}</h5><h2>Comments from Survey Respondents</h2>
    <h4>Exploration and Production Firms</h4><ul><li>We expect oil prices to remain near current levels.</li><li><p>Our crude oil well drilling plans will depend on petroleum prices.</p></li></ul></div>
    <div id='des-report-print' class='visible-print'><p>BADPRINT {prose}</p></div>
    <div class='dal-info__container'><p>BADFOOTER contact January 1, 2026.</p></div>
    </div></div></html>""".encode()


@pytest.mark.parametrize("url", [
    "http://www.dallasfed.org/research/surveys/des/2019/1901",
    "https://www.dallasfed.org.evil.test/research/surveys/des/2019/1901",
    "https://user@www.dallasfed.org/research/surveys/des/2019/1901",
    "https://www.dallasfed.org:444/research/surveys/des/2019/1901",
    "https://www.dallasfed.org/research/surveys/des/2019/2001",
    "https://www.dallasfed.org/research/surveys/des/2019/1905",
    "https://www.dallasfed.org/research/surveys/des/2017/1701",
    "https://www.dallasfed.org/research/surveys/des/2027/2701",
    "https://www.dallasfed.org/research/surveys/des/2026/2601/2602update",
    "https://www.dallasfed.org/research/surveys/des/2019/1901?version=2",
    "https://www.dallasfed.org/research/surveys/des/2019/1901#comments",
    "https://www.dallasfed.org/-/media/documents/research/surveys/des/2018/1801/des1801.pdf",
    "https://www.dallasfed.org/research/economics/2024/0701",
])
def test_strict_official_survey_routes(url):
    with pytest.raises(ValueError):
        mod.admitted_dallas_route(url)


def test_admit_matching_update_only():
    assert mod.admitted_dallas_route("https://www.dallasfed.org/research/surveys/des/2026/2601/2601update") == "https://www.dallasfed.org/research/surveys/des/2026/2601/2601update"


def test_archive_uses_actual_links_not_missing_generated_quarters():
    raw = b"<div id='content'><div class='dal-main-content'><a href='/research/surveys/des/2019/1901'>First quarter</a><a href='/research/surveys/des/2026/2603'>Third quarter</a><a href='https://evil.test/research/surveys/des/2019/1904'>Fourth quarter</a></div></div>"
    rows, excluded = mod.linked_reports(raw)
    assert len(rows) == 2 and excluded == {"not_admitted_quarterly_html_route": 1}
    assert all("published_date" not in row and "available_by" not in row for row in rows)
    assert not any(row["edition_code"] == "1902" for row in rows)


def test_article_native_exact_date_title_and_question_bound_response():
    parsed = mod.parse_dallas_body(survey())
    assert parsed["published_date"] == "2019-03-27"
    assert parsed["title"] == "Oil and Gas Activity Grows Modestly as Oil Price Jumps"
    assert parsed["survey_period"] == "First Quarter"
    assert "January 1, 2026" not in parsed["text"]
    assert "Next release" not in parsed["text"]
    assert "BADPRINT" not in parsed["text"] and "BADFOOTER" not in parsed["text"]
    assert any("What WTI" in block and "$50 per barrel" in block for block in parsed["blocks"])
    comments = [block for block in parsed["structured_blocks"] if block["evidence_type"] == "survey_respondent_comment"]
    assert len(comments) == 2
    assert "Comments from Survey Respondents" in comments[0]["text"]
    assert "Our crude oil well drilling plans" in comments[1]["text"]
    assert "First Quarter | March 27, 2019" not in parsed["scope_text"]
    assert parsed["title"] not in parsed["scope_text"]
    assert "What WTI" not in parsed["scope_text"]
    assert "Our crude oil well drilling plans" in parsed["scope_text"]


def test_table_chart_and_print_duplicates_are_excluded_actual_subtrees():
    extra = "<table><tr><td><p>BADTABLE 99.9 million barrels per day</p></td></tr></table><p class='chart-full'><img alt='BADCHART' src='chart.svg'>BADCHART</p><figure><p>BADFIGURE</p></figure><p class='text-center'>BADCHARTLINK</p><div class='visible-print'><p>BADPRINTDUP</p></div>"
    parsed = mod.parse_dallas_body(survey(extra=extra))
    assert "BAD" not in parsed["text"]
    assert parsed["removed_actual_table_subtrees"] == 1


def test_observed_modern_div_date_preserves_quarter_context():
    raw = survey().replace(b"<h5 class='survey-date'>First Quarter | March 27, 2019</h5>", b"<div class='survey-date--des'>First Quarter | March 27, 2019</div>")
    raw = raw.replace(b"<h5 id='des-reportdate'>First Quarter | March 27, 2019</h5>", b"<div class='report-date--des'>First Quarter | March 27, 2019</div>")
    parsed = mod.parse_dallas_body(raw)
    assert parsed["published_date"] == "2019-03-27"
    assert "First Quarter | March 27, 2019" in parsed["text"]


def test_title_oil_with_electricity_only_actual_body_is_rejected():
    body = " ".join(["Electricity generation increased as solar power capacity expanded and households installed more batteries in 2019."] * 12)
    raw = f"<div id='content'><div class='dal-main-content'><div id='des-report'><h5 class='survey-date'>First Quarter | March 27, 2019</h5><h2>Crude oil production and petroleum demand</h2><p>{body}</p></div></div></div>".encode()
    with pytest.raises(ValueError, match="mineral-oil prose"):
        mod.parse_dallas_body(raw)


def test_no_missing_header_date_fallback_to_body_or_footer():
    with pytest.raises(ValueError, match="header"):
        mod.parse_dallas_body(survey(header=False))


@pytest.mark.parametrize("day", ["First Quarter | 2019", "First Quarter | March 2019", "Third Quarter | October 6, 2026"])
def test_unknown_or_out_of_window_publication_fails_closed(day):
    with pytest.raises(ValueError):
        mod.parse_dallas_body(survey(day=day))


@pytest.mark.parametrize("revision", ["Updated: April 4, 2019", "Correction: This report was revised."])
def test_later_or_undated_revision_not_backdated(revision):
    with pytest.raises(ValueError, match="revision"):
        mod.parse_dallas_body(survey(extra=f"<p>{revision}</p>"))


def test_title_mismatch_fail_closed():
    raw = survey().replace(b"<h2>Oil and Gas Activity Grows Modestly as Oil Price Jumps</h2>", b"<h2>A different body headline</h2>")
    with pytest.raises(ValueError, match="headline"):
        mod.parse_dallas_body(raw)


def test_fixed_edition_spread_pilots_not_selected_by_yield():
    rows = [dict(url=f"x{i}", edition_year=str(2019 + i // 4), edition_code=f"{19 + i // 4}0{i % 4 + 1}") for i in range(31)]
    chosen = mod.select_pilots(rows, 8)
    assert len(chosen) == 8 and chosen[0]["url"] == "x0" and chosen[-1]["url"] == "x30"
    assert chosen == mod.select_pilots(list(reversed(rows)), 8)


def test_frozen_output_and_bounds(tmp_path):
    with pytest.raises(ValueError, match="six-to-twelve"):
        mod.discover(tmp_path, 1)
    (tmp_path / "source_candidates.jsonl").write_text("frozen", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        mod.discover(tmp_path)
