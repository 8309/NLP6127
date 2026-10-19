"""Uniform segmentation from verified source pages, never from old passages.

Source-specific extraction remains appropriate (HTML body versus PDF chapter),
but every admitted page uses the same sentence policy and rejection accounting.
No live corpus is overwritten by this builder.
"""
from __future__ import annotations

from collections import Counter
from datetime import date
import json
from pathlib import Path
import re

from .corpus import (REGIONS, NEW_PDF_SERIES, clean_page, extraction_suffix,
                     has_table_heading, is_prose_page, is_reference_page, split_passages)
from .store import DATA, config, digest, read_jsonl, words, write_json, write_jsonl

PROVENANCE = ['document_id', 'source', 'title', 'url', 'published_date',
              'available_by', 'date_precision', 'date_source', 'event_date',
              'event_status', 'redistribution']
OPTIONAL_PROVENANCE = ['evidence_type', 'survey_period', 'survey_release_header',
                       'publication_version_caution']


def optional_provenance(record):
    """Retain exact source context, not labels; legacy records gain no keys."""
    return {key: record[key] for key in OPTIONAL_PROVENANCE
            if isinstance(record.get(key), str) and record[key].strip()}


def source_native_context(doc, raw, text):
    """Map verified native blocks, without guessed search matches or new text."""
    if 'source_block_attribution' not in doc:
        return {}
    blocks = doc['source_block_attribution']
    if (not isinstance(blocks, list) or not blocks or any(not isinstance(block, dict)
            or not isinstance(block.get('text'), str) or not block['text'].strip() for block in blocks)):
        raise ValueError('Malformed source-native block attribution')
    if raw != '\n\n'.join(block['text'] for block in blocks):
        raise ValueError('Source-native blocks do not reproduce the exact extraction page')
    cleaned = [(index, block, clean_source_page(block['text'], doc)[0]) for index, block in enumerate(blocks)]
    cleaned = [(index, block, value) for index, block, value in cleaned if value.strip()]
    if '\n\n'.join(value for _, _, value in cleaned) != text:
        raise ValueError('Source-native block cleaning does not reproduce the exact clean page')
    mapped, boundaries, cursor, previous_independent = [], [], 0, False
    for index, block, value in cleaned:
        independent = block.get('evidence_type') == 'survey_respondent_comment' or block.get('independent_source_block') is True
        if cursor and (independent or previous_independent):
            boundaries.append(cursor)
        context = dict(source_block_index=index, char_start=cursor, char_end=cursor + len(value))
        context.update({key: block[key] for key in ['section_id', 'evidence_type', 'survey_release_header', 'question']
                        if isinstance(block.get(key), str) and block[key].strip()})
        context['independent_source_block'] = independent
        mapped.append(context)
        cursor += len(value) + 2
        previous_independent = independent
    return dict(source_native_blocks=mapped, source_hard_boundaries=boundaries)


def passage_source_context(page, start, end):
    if 'source_native_blocks' not in page:
        return {}
    return dict(source_block_attribution=[dict(block) for block in page['source_native_blocks']
                if block['char_start'] < end and start < block['char_end']])


def source_question_dependencies(page):
    """Use actual contiguous native question text, never metadata as an answer.

    A temporal target omitted by a following answer must remain in the indexed
    text. All same-question blocks form one exact interval, without crossing an
    independent respondent. Missing/overlong context is quarantined by the
    uniform segmenter; no question/year is copied or inferred into the body.
    """
    blocks = page.get('source_native_blocks', [])
    text = page['text']
    required = []
    cursor = 0
    relative = re.compile(r'\b(?:next|coming|following|previous|past|last)\s+(?:\d+\s+)?(?:months?|quarters?|years?)\b|\b(?:year[- ]end|end\s+of\s+(?:the\s+)?year)\b', re.I)
    while cursor < len(blocks):
        first = blocks[cursor]
        question = first.get('question')
        if first.get('independent_source_block') or not isinstance(question, str) or '?' not in question:
            cursor += 1
            continue
        question = clean_page(question)
        stop = cursor + 1
        while (stop < len(blocks) and not blocks[stop].get('independent_source_block')
               and isinstance(blocks[stop].get('question'), str)
               and clean_page(blocks[stop]['question']) == question):
            stop += 1
        group = blocks[cursor:stop]
        years = set(re.findall(r'\b(?:18|19|20)\d{2}\b', question))
        temporal = bool(years or relative.search(question))
        values = [text[block['char_start']:block['char_end']] for block in group]
        local_question = question in values[0]
        missing_target = (any(any(not re.search(r'\b' + year + r'\b', value) for year in years) for value in values)
                          if years else bool(relative.search(question) and any(question not in value for value in values)))
        if temporal and (not local_question or missing_target):
            required.append(dict(start=first['char_start'], end=group[-1]['char_end'], question=question,
                                 question_is_local=local_question))
        cursor = stop
    return required


def clean_source_page(raw, doc=None):
    """Remove identified furniture, not reconstruct tables or supply values.

The resulting text is frozen separately; passage offsets reference this exact
cleaned page, while extraction/raw hashes retain the source transformation.
"""
    retained, removed = [], []
    lines = raw.splitlines()
    for i, line in enumerate(lines):
        text = re.sub(r'\s+', ' ', line).strip()
        reason = None
        if not text:
            if retained and retained[-1] != '':
                retained.append('')
            continue
        if re.fullmatch(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+20\d{2}', text, re.I):
            reason = 'standalone_report_month'
        elif re.search(r'^(?:U\.S\. Energy Information Administration\s*\||Strategic Petroleum Reserve Annual Report.*\|\s*Page|USDOI Economic Analysis Methodology.*BOEM)', text, re.I):
            reason = 'running_header'
        elif re.fullmatch(r'(?:Summary|Chapter\s+\d+\s*:[^|]+)\s+(?:[ivxlcdm]+|\d+)\s+USDOI\s*\|\s*BOEM', text, re.I):
            # Raw PDF order can put the real bottom-margin label before the
            # body. Remove only the entire observed agency/page footer line;
            # retain actual Summary/Background headings and all body prose.
            reason = 'running_page_footer'
        elif re.fullmatch(r'(?:Overview|Global oil markets|Oil prices|Crude oil prices|Liquid fuels|Petroleum and other liquids|Supply and consumption|International petroleum|U\.S\. petroleum|World oil markets|Petroleum|Oil|Contents|Table of Contents)', text, re.I):
            reason = 'standalone_section_heading'
        elif re.fullmatch(r'(?:Short-Term Energy Outlook\s*(?:\(STEO\))?|Forecast highlights|Winter Fuels Outlook|Global liquid fuels|Recent Actions|Body)', text, re.I):
            reason = 'standalone_section_heading'
        elif re.fullmatch(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+20\d{2}', text, re.I):
            reason = 'standalone_publication_date'
        elif doc and text == doc.get('title'):
            reason = 'duplicate_article_title'
        elif re.match(r'^(?:Data source|Source)\s*:', text, re.I):
            reason = 'source_credit_metadata'
        elif re.match(r'^(?:\d+(?:\.\d+){1,4}\s+[A-Z]|Chart\s+\d+(?:\.\d+)?\s+)', text) and not re.search(r'\b(?:is|are|was|were|has|have|will)\b', text):
            reason = 'standalone_section_or_chart_heading'
        elif re.match(r'^(?:Principal contributors?\s*:|Tags\s*:)', text, re.I):
            removed.extend({'line': j + 1, 'text': t, 'reason': 'article_footer'} for j, t in enumerate(lines[i:], i) if t.strip())
            break
        elif re.match(r'^For questions about This Week in Petroleum, contact', text, re.I):
            removed.extend({'line': j + 1, 'text': t, 'reason': 'contact_footer'} for j, t in enumerate(lines[i:], i) if t.strip())
            break
        # HTML get_text(newline) can place an inline linked report title on its
        # own line. It is not furniture when the neighbouring lines visibly
        # continue the same clause (e.g. "In the June [STEO], we forecast...").
        previous = lines[i - 1].strip() if i else ''
        following = lines[i + 1].strip() if i + 1 < len(lines) else ''
        inline_clause = (previous and not re.search(r'[.!?:;]\s*$', previous)
                         and bool(re.match(r'^[,;)]|^[a-z]', following)))
        if inline_clause and reason in {'standalone_section_heading', 'duplicate_article_title',
                                        'standalone_report_month', 'standalone_publication_date'}:
            reason = None
        if reason:
            removed.append({'line': i + 1, 'text': text, 'reason': reason})
            if retained and retained[-1] != '':
                retained.append('')
        else:
            # Bullet glyphs delimit paragraphs, not an oil-fact token. Do not
            # remove a glyph inside a clause (a likely extraction error).
            retained.append(re.sub(r'^\s*[•●▪]\s*', '', text))
    # Retain observed blank/section boundaries, but single PDF line wraps remain
    # inside a paragraph. Never append invented punctuation to a heading/tail.
    blocks = re.split(r'\n\s*\n', '\n'.join(retained))
    return '\n\n'.join(clean_page(block) for block in blocks if clean_page(block).strip()), removed


def _native_appendix_cover(raw):
    """A whole-page native cover, not a TOC entry or prose cross-reference."""
    match = re.fullmatch(r'\s*APPENDIX\s+([A-Z])\s*\n([\s\S]+?)\s*', raw)
    if not match:
        return None
    title = re.sub(r'\s+', ' ', match[2]).strip()
    # Native covers contain only the uppercase appendix title. A following
    # body paragraph, bibliographic list or TOC dot-leader is not a cover.
    if not title.isupper() or re.search(r'[.!?]|\.{2,}|\d', title):
        return None
    return dict(letter=match[1], title=title)


def native_section_exclusions(raw_pages, is_pdf=True):
    """Hold entire bibliography or public-comment/response appendices.

    Exact cover titles and every subsequent native letter-page running header
    establish a source section. No passage IDs, publisher IDs, fixed physical
    numbers or word-size thresholds are used. A strong section cue with broken
    pagination is held to the next actual different appendix cover (or EOF),
    with failed validation recorded, never silently admitted as prose.
    """
    if not is_pdf:
        return {}
    covers = [(number, cover) for number, raw in enumerate(raw_pages, 1)
              if (cover := _native_appendix_cover(raw)) is not None]
    output = {}
    for index, (first, cover) in enumerate(covers):
        title, letter = cover['title'], cover['letter']
        kind = ('references' if re.fullmatch(r'REFERENCES|BIBLIOGRAPHY|LITERATURE CITED', title) else
                'public_comments_and_responses' if re.search(r'\b(?:RESPONSES? TO (?:PUBLIC )?COMMENTS?|PUBLIC COMMENTS? AND RESPONSES?)\b', title) else None)
        if kind is None:
            continue
        following = next(((number, value) for number, value in covers[index+1:] if value['letter'] != letter), None)
        last = following[0] - 1 if following else len(raw_pages)
        title_pattern = re.escape(title).replace(r'\ ', r'\s+')
        header = re.compile(r'^\s*' + title_pattern + r'\s+' + letter + r'[-\u2013]\s*(\d+)\s*$', re.I)
        failures, observed = [], {}
        for number in range(first + 1, last + 1):
            line = next((line.strip() for line in raw_pages[number-1].splitlines() if line.strip()), '')
            match = header.fullmatch(line)
            printed = int(match[1]) if match else None
            observed[number] = dict(native_running_header=line, native_printed_page=printed)
            if printed != number - first + 1:
                failures.append(dict(physical_page=number, expected_printed_page=number-first+1,
                                     actual_printed_page=printed, reason='missing_or_noncontinuous_native_running_header'))
        if following and ord(following[1]['letter']) != ord(letter) + 1:
            failures.append(dict(physical_page=following[0], reason='unexpected_next_native_appendix_letter',
                                 actual_letter=following[1]['letter']))
        if last == first:
            failures.append(dict(physical_page=first, reason='native_section_has_no_verified_running_pages'))
        validation = 'failed_closed_section_hold' if failures else 'verified_continuous_native_running_headers'
        reason = 'native_' + kind + '_appendix' + ('_boundary_validation_failed' if failures else '')
        proof = dict(kind=kind, appendix_letter=letter, native_cover_title=title, first_physical_page=first,
                     last_physical_page=last, cover_raw_page_sha256=digest(raw_pages[first-1]),
                     next_appendix_physical_page=following[0] if following else None,
                     next_appendix_letter=following[1]['letter'] if following else None,
                     next_cover_raw_page_sha256=digest(raw_pages[following[0]-1]) if following else None,
                     validation=validation, validation_errors=failures)
        for number in range(first, last + 1):
            output[number] = dict(reason=reason, native_section=dict(proof, **observed.get(number, {})))
    # A bibliography can also be the final numbered subsection of a substantive
    # appendix. Its first page may still contain valid preceding prose; hold
    # that whole mixed page rather than infer a layout-dependent cut point.
    reference_heading = re.compile(r'(?m)^\s*([A-Z](?:\.\d+)+)\s+(REFERENCES|BIBLIOGRAPHY|LITERATURE CITED)\s*$')
    numbered_heading = re.compile(r'(?m)^\s*([A-Z](?:\.\d+)+)\s+([A-Z][A-Z\s&(),:/\u2013\u2014-]*?)\s*$')
    running = re.compile(r'^\s*(.+?)\s+([A-Z])[-\u2013]\s*(\d+)\s*$')
    for first, raw in enumerate(raw_pages, 1):
        heading = reference_heading.search(raw)
        if first in output or heading is None:
            continue
        label, title = heading[1], heading[2]
        letter = label.split('.')[0]
        label_parts = [int(part) for part in label.split('.')[1:]]
        first_line = next((line.strip() for line in raw.splitlines() if line.strip()), '')
        first_header = running.fullmatch(first_line)
        native_title = first_header[1] if first_header and first_header[2] == letter else None
        printed_start = int(first_header[3]) if native_title else None
        following = next(((number, value) for number, value in covers if number > first and value['letter'] != letter), None)
        last = following[0] - 1 if following else len(raw_pages)
        boundary_kind = 'next_native_appendix_cover' if following else 'physical_eof_or_unverified_tail'
        boundary_heading = None
        for number in range(first + 1, last + 1):
            peer = next((match for match in numbered_heading.finditer(raw_pages[number-1])
                         if match[1].split('.')[0] == letter and match[2].strip() not in {'REFERENCES', 'BIBLIOGRAPHY', 'LITERATURE CITED'}
                         and len(match[1].split('.')) <= len(label.split('.'))
                         and [int(part) for part in match[1].split('.')[1:]] > label_parts), None)
            if peer:
                last, boundary_kind, boundary_heading = number - 1, 'next_native_peer_heading', peer.group().strip()
                break
        failures, observed = [], {}
        for number in range(first, last + 1):
            line = next((line.strip() for line in raw_pages[number-1].splitlines() if line.strip()), '')
            match = running.fullmatch(line)
            printed = int(match[3]) if match and match[2] == letter and match[1] == native_title else None
            observed[number] = dict(native_running_header=line, native_printed_page=printed)
            expected = printed_start + number - first if printed_start is not None else None
            if printed is None or printed != expected:
                failures.append(dict(physical_page=number, expected_printed_page=expected, actual_printed_page=printed,
                                     reason='missing_or_noncontinuous_native_running_header'))
        proof = dict(kind='reference_subsection', appendix_letter=letter, native_reference_heading=heading.group().strip(),
                     native_section_label=label, native_section_title=title, native_cover_title=native_title,
                     first_physical_page=first, last_physical_page=last, heading_start_offset=heading.start(),
                     start_raw_page_sha256=digest(raw), whole_page_conservative_hold=True,
                     end_boundary_kind=boundary_kind, next_native_peer_heading=boundary_heading,
                     next_appendix_physical_page=following[0] if following and boundary_kind == 'next_native_appendix_cover' else None,
                     validation='failed_closed_section_hold' if failures else 'verified_continuous_native_running_headers',
                     validation_errors=failures)
        reason = 'native_reference_subsection' + ('_boundary_validation_failed' if failures else '')
        for number in range(first, last + 1):
            if number not in output:
                output[number] = dict(reason=reason, native_section=dict(proof, **observed[number]))
    return output


def page_exclusion(doc, raw, number, is_pdf, native_sections=None):
    if native_sections and number in native_sections:
        return native_sections[number]['reason']
    if str(number) in doc.get('excluded_physical_pages', {}):
        return 'recorded_physical_page_exclusion'
    if not raw.strip():
        return 'blank_or_outside_verified_chapter'
    if is_pdf and re.match(r'\A\s*(?:References?|Bibliography|Literature Cited)\s+(?:[A-Z][.-]\s*)?(?:\d+|[ivxlcdm]+)\s*(?:\n|$)', raw, re.I):
        return 'native_reference_running_header_page'
    if is_pdf and re.match(r'\A\s*Responses? to (?:Public )?Comments?[^\n]*\s+[A-Z][-\u2013]\d+\s*(?:\n|$)', raw, re.I):
        return 'native_public_comment_response_running_header_page'
    margin = re.sub(r'\s+', '', raw[:250]).lower()
    if doc['source'] == 'WB-CMO' and any(label in margin for label in ['agriculturecommodity', 'metalsandmineralscommodity', 'fertilizerscommodity', 'preciousmetalscommodity']):
        return 'explicit_non_oil_commodity_chapter'
    # Historical manual exclusions used the old 200/65 text fingerprint. New
    # IDs cannot erase those decisions; hold the affected source page until the
    # excluded fragment can be independently repaired/isolated.
    if doc.get('excluded_passage_ids'):
        for old_text in split_passages(clean_page(raw), target=200, minimum=65):
            old_id = doc['document_id'] + '-' + digest(' '.join(words(old_text)))[:16]
            if old_id in doc['excluded_passage_ids']:
                return 'recorded_legacy_passage_exclusion_requires_repair'
    if is_reference_page(raw):
        return 'reference_page'
    if re.search(r'\bDescription\s+of\s+price\s+series\b', raw, re.I):
        return 'mixed_price_series_dictionary_pending_structured_extraction'
    if is_pdf and re.search(
        r'\b(?:pr\s+ices|m\s+ain|t\s+he|out\s+put|economi\s+es|commoditi\s+es|'
        r'Seve\s+n|po\s+ols|des\s+igned|T\s+exas|Septem\s+ber|'
        r'Liby\s+a|Nigeri\s+a|rejo\s+ined)\b', raw, re.I
    ):
        # Observable fractured words signal damaged PDF layout; do not guess
        # repairs or let an incidental oil comparator admit interleaved prose.
        # The same source-level rule covers fractured body/footnote tokens;
        # a rendered source page, not retrieval performance, motivates these.
        return 'fragmented_word_extraction_pending_layout_repair'
    if has_table_heading(raw):
        return 'mixed_table_page_not_reconstructed'
    if is_pdf and re.search(r'(?im)^\s*(?:FIGURE|Chart)\s+(?:[A-Z]{1,3}[.-])?\d', raw):
        return 'mixed_figure_page_pending_layout_aware_extraction'
    if re.search(r'(?i)\b(?:index|equation)\s+is\s+of\s+the\s+form\s*:', raw):
        return 'formula_page_pending_structured_extraction'
    if doc['source'] == 'WB-CMO' and re.search(r'Asymmetric impacts of metal price shocks|aggregate results mask asymmetric impacts of metal', raw, re.I):
        return 'explicit_non_oil_study_section'
    if re.search(r'U\.S\. energy market indicators', raw, re.I):
        return 'flattened_overview_indicator_table'
    if re.search(r'(?im)^\s*(?:abbreviations\s*(?:and|&)\s*acronyms|variable\s+definitions|data\s+dictionary)\b', raw):
        return 'dictionary_page'
    if re.search(r'Variable\s+Description\s+Unit\s+Source|LFMM\s+variable\s+Definition\s+API\s+keys', raw, re.I):
        return 'input_variable_table'
    if doc['source'] == 'UK-ENERGY-TRENDS' and re.search(r'About\s+this\s+release', raw, re.I) and re.search(r'In\s+this\s+release', raw, re.I):
        return 'navigation_cover'
    if doc['source'] == 'UK-DUKES' and re.search(r'\b(?:19|20)\d{3}\b', raw):
        return 'ambiguous_year_superscript'
    if is_pdf and not is_prose_page(raw):
        return 'numeric_or_insufficient_prose_page'
    return None


def extract_existing(doc, data):
    """Use source-bound caches; missing recognized HTML bodies can be re-read.

Missing PDF caches fail closed rather than silently changing the extraction
tool/version. PDF repair is a separate audited workflow.
"""
    raw_path = (data / doc['raw_path']).resolve()
    if not raw_path.is_relative_to(data.resolve()):
        raise ValueError('Raw path escapes data')
    raw_bytes = raw_path.read_bytes()
    if digest(raw_bytes) != doc['raw_sha256']:
        raise ValueError('Raw source hash mismatch')
    if not raw_bytes.startswith(b'%PDF') and doc['source'] == 'BSEE-HURRICANE' and b'<table' in raw_bytes.lower():
        # Table structure is actually available in HTML. Exclude that subtree,
        # rather than infer columns from an old flattened-body cache.
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(raw_bytes, 'html.parser')
        main = soup.select_one('main article .field--name-body')
        if main is None:
            raise ValueError('No recognized article body')
        removed = len(main.select('table'))
        for element in main.select('table,.usa-alert,figure,script,style'):
            element.decompose()
        body = main.get_text('\n', strip=True)
        return [body], dict(extraction_path=None, extraction_sha256=digest(body),
                            extraction_method='recognized_html_body_tables_excluded_v2', removed_html_tables=removed)
    cache = data / 'extracted' / (doc['raw_sha256'] + extraction_suffix(doc['source']))
    if cache.exists():
        pages = json.loads(cache.read_text(encoding='utf-8'))
        if not isinstance(pages, list) or not all(isinstance(p, str) for p in pages):
            raise ValueError('Malformed extraction cache')
        return pages, dict(extraction_path=cache.relative_to(data).as_posix(), extraction_sha256=digest(cache.read_bytes()))
    if raw_bytes.startswith(b'%PDF'):
        raise ValueError('Missing verified PDF extraction cache; separate repair required')
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(raw_bytes, 'html.parser')
    for element in soup.select('script,style,nav,header,footer'):
        element.decompose()
    main = (soup.select_one('main article .field--name-body') if doc['source'] == 'BSEE-HURRICANE' else
            soup.select_one('.tie-article') or soup.select_one('.l-col.article') or
            soup.select_one('.pagecontent.print .main_col') or
            soup.select_one('article .field--name-field-body') or soup.select_one('article') or soup.select_one('main'))
    if main is None:
        raise ValueError('No recognized article body')
    for element in main.select('.usa-alert,figure'):
        element.decompose()
    body = main.get_text('\n', strip=True)
    if doc['source'] == 'OFAC':
        body = re.split(r'Specially Designated Nationals List Updates', body, flags=re.I)[0]
    return [body], dict(extraction_path=None, extraction_sha256=digest(body), extraction_method='recognized_html_body_v1')


def build_unified_corpus(output_dir, data_dir=None, additional_documents=None):
    from .segmentation import SegmentationPolicy, segment_page
    data, output = Path(data_dir or DATA), Path(output_dir)
    if output.resolve() == data.resolve():
        raise ValueError('Uniform rebuild must be staged; use guarded publication')
    settings = config()['corpus']
    policy = SegmentationPolicy(target_words=settings.get('chunk_words', 160), minimum_words=settings.get('min_words', 65),
                                maximum_words=settings.get('max_words', 240), maximum_context_words=settings.get('max_context_words', 480))
    pages, passages, quarantine, exclusions, errors = [], [], [], [], []
    seen = {}
    documents = read_jsonl(data / 'documents.jsonl') + list(additional_documents or [])
    if len({d['document_id'] for d in documents}) != len(documents) or len({d['url'] for d in documents}) != len(documents):
        raise ValueError('Supplemental documents must be new, nonduplicate source URLs/IDs')
    for n, doc in enumerate(documents, 1):
        reason = doc.get('ingest_exclusion_reason')
        if doc['source'] == 'EIA-FINANCE':
            reason = reason or 'catalogue_source_pending_factual_extraction'
        if reason:
            exclusions.append(dict(document_id=doc['document_id'], scope='document', reason=reason))
            continue
        try:
            available = doc.get('available_by')
            if not available or not settings['start_date'] <= available <= settings['end_date']:
                raise ValueError('Publication availability absent or outside fixed window')
            date.fromisoformat(available)
            raw_pages, extraction = extract_existing(doc, data)
            native_sections = native_section_exclusions(raw_pages, doc['raw_path'].endswith('.pdf'))
            if 'source_block_attribution' in doc and len(raw_pages) != 1:
                raise ValueError('Source-native attribution requires a single verified extraction page')
            for number, raw in enumerate(raw_pages, 1):
                reason = page_exclusion(doc, raw, number, doc['raw_path'].endswith('.pdf'), native_sections)
                if reason:
                    record = dict(document_id=doc['document_id'], page=number, scope='page', reason=reason,
                                  raw_page_sha256=digest(raw), raw_page_words=len(words(raw)))
                    if number in native_sections:
                        record['native_section'] = native_sections[number]['native_section']
                    exclusions.append(record)
                    continue
                text, removed = clean_source_page(raw, doc)
                if not text.strip():
                    exclusions.append(dict(document_id=doc['document_id'], page=number, scope='page', reason='empty_after_furniture_cleaning'))
                    continue
                source_page_id = f"{doc['document_id']}-page-{number}-{digest(text)[:16]}"
                page = {key: doc.get(key) for key in PROVENANCE}
                page.update(optional_provenance(doc))
                page.update(source_page_id=source_page_id, page=number, text=text, word_count=len(words(text)),
                            raw_sha256=doc['raw_sha256'], raw_path=doc['raw_path'], raw_page_sha256=digest(raw),
                            cleaning_removed_lines=removed, **extraction)
                page.update(source_native_context(doc, raw, text))
                pages.append(page)
                result = (segment_page(text, policy, hard_boundaries=page['source_hard_boundaries'],
                                       required_contexts=source_question_dependencies(page))
                          if 'source_hard_boundaries' in page else segment_page(text, policy))
                for rejected in result['quarantined']:
                    quarantine.append(dict(source_page_id=source_page_id, document_id=doc['document_id'], page=number,
                                           char_start=rejected['start'], char_end=rejected['end'], text=rejected['text'],
                                           word_count=rejected['word_count'], reason=rejected['reason']))
                for accepted in result['accepted']:
                    fingerprint = digest(' '.join(words(accepted['text'])))
                    if fingerprint in seen:
                        quarantine.append(dict(source_page_id=source_page_id, document_id=doc['document_id'], page=number,
                                               char_start=accepted['start'], char_end=accepted['end'], text=accepted['text'],
                                               word_count=accepted['word_count'], reason='exact_normalized_duplicate',
                                               duplicate_of=seen[fingerprint]))
                        continue
                    row = {key: doc.get(key) for key in PROVENANCE}
                    row.update(optional_provenance(doc))
                    row.update(passage_id=doc['document_id']+'-u-'+digest(f"{source_page_id}:{accepted['start']}:{accepted['end']}:{accepted['text']}")[:24],
                               source_page_id=source_page_id, page=number, text=accepted['text'], word_count=accepted['word_count'],
                               char_start=accepted['start'], char_end=accepted['end'], segmentation=result['policy'],
                               segmentation_exceptions=accepted.get('exceptions', []),
                               regions=[name for name, pattern in REGIONS.items() if re.search(pattern, accepted['text'], re.I)],
                               region_method='keyword_heuristic')
                    row.update(passage_source_context(page, accepted['start'], accepted['end']))
                    passages.append(row)
                    seen[fingerprint] = row['passage_id']
        except Exception as exc:
            errors.append(dict(document_id=doc['document_id'], error=str(exc)))
        if n % 100 == 0:
            print(f'Uniform source build {n}/{len(documents)}: {len(passages)} accepted, {len(quarantine)} quarantined', flush=True)
    write_jsonl(output / 'source_pages.jsonl', pages)
    write_jsonl(output / 'passages.jsonl', passages)
    write_jsonl(output / 'quarantine.jsonl', quarantine)
    write_jsonl(output / 'source_exclusions.jsonl', exclusions)
    vocabulary = Counter(w for p in passages for w in words(p['text']))
    counts = [p['word_count'] for p in passages]
    import statistics
    report = dict(version='semantic-page-v1', segmentation_version='semantic-page-v1',
                  passages=len(passages), documents=len({p['document_id'] for p in passages}),
                  words=sum(vocabulary.values()), unique_words=len(vocabulary), source_pages=len(pages),
                  by_source=dict(Counter(p['source'] for p in passages)),
                  by_year=dict(Counter(p['available_by'][:4] for p in passages)),
                  date_precision=dict(Counter(p['date_precision'] for p in passages)),
                  corpus_sha256=digest((output / 'passages.jsonl').read_bytes()),
                  source_pages_sha256=digest((output / 'source_pages.jsonl').read_bytes()),
                  quarantine_sha256=digest((output / 'quarantine.jsonl').read_bytes()),
                  source_exclusions_sha256=digest((output / 'source_exclusions.jsonl').read_bytes()),
                  source_page_words=sum(p['word_count'] for p in pages), quarantined_words=sum(p['word_count'] for p in quarantine),
                  quarantined_units=len(quarantine), quarantine_reasons=dict(Counter(p['reason'] for p in quarantine)),
                  source_exclusion_reasons=dict(Counter(e['reason'] for e in exclusions)), extraction_errors=errors,
                  chunk_policy=result['policy'] if pages else {},
                  word_length=dict(min=min(counts, default=0), median=statistics.median(counts) if counts else 0, max=max(counts, default=0)),
                  exception_counts=dict(Counter(e for p in passages for e in p['segmentation_exceptions'])),
                  exact_deduplication=True, overlap_words=0, old_passage_fallback=False,
                  labels_coverage=0, archived_machine_labels=len(read_jsonl(data / 'annotations/passage_labels.jsonl')),
                  unlabelled_corpus_passages=len(passages),
                  meets_size_requirement=len(passages) >= 10000 and sum(vocabulary.values()) >= 100000,
                  external_model_calls=0, human_labels_created=0,
                  limitations=['Structural sentence policy is not exhaustive semantic or human approval.',
                               'PDF page boundaries are not crossed; incomplete page-edge clauses are quarantined.',
                               'Tables are excluded, not reconstructed; exact deduplication is not semantic deduplication.',
                               'Prior labels remain bound to the archived original text, never new IDs.'])
    write_json(output / 'reports/corpus.json', report)
    return report
