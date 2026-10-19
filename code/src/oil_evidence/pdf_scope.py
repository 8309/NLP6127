"""Fail-closed oil chapter scoping for mixed-topic official PDF series.

Physical page numbers are one-based, not report-printed page labels. Source
document text remains unchanged; the caller owns extraction and corpus writes.
"""

import re


_AU_CHAPTERS = (
    "Executive summary", "Overview", "Macroeconomic outlook", "Iron ore",
    "Metallurgical coal", "Thermal coal", "Gas", "Oil", "Uranium", "Gold",
    "Aluminium, alumina and bauxite", "Aluminium", "Copper", "Nickel", "Zinc",
    "Lithium", "Other critical minerals", "Principal markets for Australia’s resource and energy exports",
    "Appendices", "Appendix", "About the edition",
)
_AU_PATTERNS = [(title.lower(), re.compile(
    r"^(?:\d{1,2}\s*\.?\s*)?" + re.escape(title).replace(r"\ ", r"\s+")
    + r"(?:\s*" + re.escape(title).replace(r"\ ", r"\s+") + r")?$", re.I))
    for title in _AU_CHAPTERS]
_UK_CHAPTER = re.compile(r"^Chapter\s+(\d{1,2})\s*[:.]\s*(.+?)\s*$", re.I)


def _lines(text):
    return [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]


def _contents_page(text):
    lines = _lines(text)
    return bool(re.search(r"\b(?:table of contents|contents|data tables in this release)\b",
                          " ".join(lines[:10]), re.I))


def _au_headings(text):
    if _contents_page(text):
        return []
    found = []
    # REQ's title is at the top of the page, after its report/page-number header.
    # Reject later chart labels and mention-only lines anywhere in body prose.
    # March 2025's Uranium cover has seven empty numbered lines before its
    # real "8. Uranium" heading (ninth nonempty extracted line).
    for line in _lines(text)[:12]:
        for title, pattern in _AU_PATTERNS:
            if pattern.fullmatch(line):
                found.append((title, line))
    return found


def _au_toc_ranges(pages):
    """Report-printed Oil range from a real front-matter contents page."""
    ranges = []
    for text in pages[:10]:
        if not _contents_page(text):
            continue
        entries = []
        for line in _lines(text):
            match = re.fullmatch(r"(.+?)\s+(\d{1,3})", line)
            if match and match[1].lower() in {title.lower() for title in _AU_CHAPTERS}:
                entries.append((match[1].lower(), int(match[2])))
        for index, (title, number) in enumerate(entries):
            if title != "oil":
                continue
            later = [value for _, value in entries[index + 1:] if value > number]
            if later:
                ranges.append((number, later[0]))
    return ranges


def _au_printed_page(text):
    match = re.search(r"Resources\s+and\s+Energy\s+Quarterly\s*(?:\|\s*)?"
                      r"[A-Za-z]+\s+20\d{2}\s+(\d{1,3})\b", text, re.I)
    return int(match[1]) if match else None


def _au_scope(pages):
    toc = _au_toc_ranges(pages)
    start_candidates = []
    for number, text in enumerate(pages, 1):
        for title, line in _au_headings(text):
            if title != "oil":
                continue
            strong = bool(re.match(r"^\d{1,2}\s*\.?\s*Oil", line, re.I))
            strong |= bool(re.search(r"Australia[’']s\s+oil\s+sector", text, re.I))
            printed = _au_printed_page(text)
            toc_confirms = any(printed == low for low, _ in toc)
            if strong or toc_confirms:
                start_candidates.append(number)
    # Ambiguous duplicate chapter starts are not silently interpreted as ranges.
    if len(start_candidates) != 1:
        return set()
    start = start_candidates[0]
    for end in range(start + 1, len(pages) + 1):
        headings = _au_headings(pages[end - 1])
        other = [title for title, _ in headings if title != "oil"]
        if other:
            # A known TOC should agree with the next actual chapter boundary.
            start_printed = _au_printed_page(pages[start - 1])
            end_printed = _au_printed_page(pages[end - 1])
            related = [(low, high) for low, high in toc if low == start_printed]
            if related and not any(high == end_printed for _, high in related):
                return set()
            return set(range(start, end))
    return set()


def _uk_headings(text):
    if _contents_page(text):
        return []
    found = []
    for line in _lines(text):
        match = _UK_CHAPTER.fullmatch(line)
        if match:
            found.append((int(match[1]), match[2].lower()))
    # Lists of chapter titles are front/back-matter navigation, not chapter starts.
    return found if len(found) == 1 else []


def _uk_scope(pages):
    allowed_titles = {"oil and oil products", "petroleum", "petroleum and petroleum products"}
    headings = [(physical, chapter, title) for physical, text in enumerate(pages, 1)
                for chapter, title in _uk_headings(text)]
    starts = [physical for physical, chapter, title in headings if chapter == 3 and title in allowed_titles]
    if len(starts) != 1:
        return set()
    start = starts[0]
    for end, chapter, title in headings:
        if end > start and chapter == 4 and title in {"natural gas", "gas"}:
            return set(range(start, end))
        if end > start and chapter > 3:
            # Missing/corrupt Chapter 4 means the actual oil endpoint is unknown.
            return set()
    return set()


def oil_page_numbers(pages: list[str], source: str) -> set[int] | None:
    """Return verified physical oil chapter pages or fail closed for mixed books.

    AU uses standalone chapter covers plus the TOC when available. UK uses
    explicit Chapter 3/4 headings anywhere in the text layer: one verified gas
    cover places its visually topmost heading at the END of extracted text.
    No boundary is inferred from oil keywords, table names or a fixed page count.
    """
    if source not in {"AU-REQ", "UK-DUKES"}:
        return None
    if not all(isinstance(page, str) for page in pages):
        raise TypeError("pages must contain raw extracted text strings")
    return _au_scope(pages) if source == "AU-REQ" else _uk_scope(pages)
