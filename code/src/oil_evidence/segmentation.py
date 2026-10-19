"""One offset-safe policy for semantic passages from cleaned source pages.

This is deliberately not a linguistic parser.  It preserves complete detected
sentences and their explicit dependent continuations; uncertain fragments stay
in quarantine rather than being repaired or silently dropped.  Source cleaning,
publication dates, exact/near deduplication and document provenance belong to
the caller.  Every non-whitespace character in the supplied page is accounted
for exactly once in an accepted or quarantined interval.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re

from .store import words


@dataclass(frozen=True)
class SegmentationPolicy:
    version: str = "semantic-page-v1"
    target_words: int = 160
    minimum_words: int = 65
    maximum_words: int = 240
    maximum_context_words: int = 480
    require_oil_scope: bool = True

    def __post_init__(self):
        for field in ("target_words", "minimum_words", "maximum_words", "maximum_context_words"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(field + " must be a positive integer")
        if not self.minimum_words <= self.target_words <= self.maximum_words <= self.maximum_context_words:
            raise ValueError("Expected minimum <= target <= maximum <= maximum_context")
        if not isinstance(self.require_oil_scope, bool):
            raise ValueError("require_oil_scope must be Boolean")
        if not isinstance(self.version, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.version):
            raise ValueError("Invalid segmentation policy version")


DEFAULT_POLICY = SegmentationPolicy()

# A final country abbreviation can also end a sentence.  A following pronoun or
# determiner is a strong boundary cue; "U.S. crude oil" is not such a cue.
_COUNTRY_ABBREVIATION = re.compile(r"\b(?:U\.\s*S|U\.\s*K|E\.\s*U)\.$", re.I)
_ABBREVIATION = re.compile(
    r"(?:\b(?:e\.g|i\.e|a\.m|p\.m|m\.b|b\.d|b\.c\.f|m\.b\.d)|"
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Fig|Figs|Eq|Eqs|No|Nos|vs|approx|"
    r"Inc|Ltd|Co|Corp|Dept|Vol|pp|p|et|al|v|Cir|Art|Sec|Ch|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)|"
    r"\b(?:[A-Z]\.){2,}[A-Z]?)\.$", re.I
)
_FINAL_BOUNDARY_CUE = re.compile(
    r"^(?:The|A|An|This|That|These|Those|It|Its|They|Their|We|Our|He|She|"
    r"However|Therefore|Meanwhile|In|By|During|After|Before|Since|Although|"
    r"As|According|Oil|Crude|Production|Output|Supply|Demand|Exports?|Imports?)\b"
)
_DEPENDENT = re.compile(
    r"^(?:It|Its|This|That|These|Those|They|Their|Such|However|Therefore|"
    r"Thus|Hence|Consequently|Accordingly|Instead|Meanwhile|Furthermore|"
    r"Moreover|Additionally|Nevertheless|Nonetheless|Otherwise|Conversely|"
    r"Also|Yet|And|But|Or|So|In\s+turn|In\s+addition|In\s+contrast|"
    r"By\s+contrast|On\s+the\s+other\s+hand|As\s+a\s+result|"
    r"For\s+instance|For\s+example|In\s+other\s+words|Similarly|"
    r"As\s+noted\s+earlier|Since\s+then|Moving\s+forward|Looking\s+ahead|"
    r"As\s+such|Over\s+the\s+same\s+period|During\s+the\s+same\s+period)\b", re.I
)
_REGIONAL_PRICE_CONTINUATION = re.compile(
    r"^(?:The\s+)?(?:East\s+Coast|West\s+Coast|Midwest|Gulf\s+Coast|"
    r"Rocky\s+Mountain|regional)\b[^.!?]{0,100}\bprices?\b", re.I
)
_INTERNAL_ANAPHOR = re.compile(
    r"\b(?:this|that|these|those|such)\s+(?:period|increase|decrease|decline|"
    r"rise|fall|change|gain|growth|difference|exports?|imports?|economies|"
    r"countries|companies|polic(?:y|ies)|measures?|forecasts?|estimates?|"
    r"price\s+levels?|production|output|supply|demand|stocks?|volumes?|projects?|refineries|facilities|"
    r"shipments?|flows?|capacity|results?|shifts?|substitutions?)\b|"
    r"\bthe\s+(?:next\s+largest|production\s+gain|outlook\s+period)\b", re.I
)
_CONTEXTUAL_SUBJECT = re.compile(
    r"^(?:The\s+)?(?:increase|decrease|decline|rise|fall|change|difference|"
    r"country|countries|company|companies|measure|measures|policy|policies|"
    r"agreement|agreements|forecast|forecasts|estimate|estimates|outlook|"
    r"production|output|supply|demand|exports?|imports?|inventor(?:y|ies)|"
    r"stocks|prices?|volumes?|shipments?|flows?|capacity|numbers?|chart|figure)\b", re.I
)
_FINITE_VERB = re.compile(
    r"\b(?:is|are|was|were|has|have|had|will|would|can|could|may|might|"
    r"shall|should|must|cannot|did|does|do|stood|saw|forecast(?:s|ed)?|project(?:s|ed)?|expect(?:s|ed)?|"
    r"estimate(?:s|d)?|increase(?:s|d)?|decrease(?:s|d)?|decline(?:s|d)?|"
    r"rise(?:s)?|rose|risen|fall(?:s)?|fell|fallen|reach(?:es|ed)?|"
    r"grow(?:s)?|grew|grown|produce(?:s|d)?|remain(?:s|ed)?|"
    r"export(?:s|ed)?|import(?:s|ed)?|account(?:s|ed)?|"
    r"include(?:s|d)?|represent(?:s|ed)?|provide(?:s|d)?|"
    r"reduce(?:s|d)?|restrict(?:s|ed)?|permit(?:s|ted)?|"
    r"designate(?:s|d)?|announce(?:s|d)?|implement(?:s|ed)?|"
    r"suspend(?:s|ed)?|impose(?:s|d)?|require(?:s|d)?|"
    r"use(?:s|d)?|report(?:s|ed)?|result(?:s|ed)?|"
    r"average(?:s|d)?|total(?:s|ed)?|exceed(?:s|ed)?|"
    r"consist(?:s|ed)?|support(?:s|ed)?|affect(?:s|ed)?|"
    r"reflect(?:s|ed)?|allow(?:s|ed)?|continue(?:s|d)?|"
    r"compare(?:s|d)?|operate(?:s|d)?|transport(?:s|ed)?|"
    r"move(?:s|d)?|deliver(?:s|ed)?|purchase(?:s|d)?|"
    r"release(?:s|d)?|store(?:s|d)?|sell(?:s)?|sold|"
    r"consum(?:e|es|ed)|process(?:es|ed)?|refine(?:s|d)?|"
    r"follow(?:s|ed)?|interrupt(?:s|ed)?|rule(?:s|d)?|"
    r"earn(?:s|ed)?|receiv(?:e|es|ed)|invest(?:s|ed)?)\b", re.I
)
_OIL = re.compile(
    r"\b(?:oil|crude|petroleum|OPEC|refiner\w*|gasoline|diesel|distillates?|"
    r"jet\s+fuel|liquid\s+fuels?|"
    r"bbls?|barrels?)\b", re.I
)
_NON_MINERAL_OIL = re.compile(
    r"\b(?:(?:soybean|soy|vegetable|palm|cooking|seed|olive|virgin|corn|"
    r"coconut|sunflower|canola|rapeseed|fish|essential)\s+oil|"
    r"oil[-\s]+equivalent|(?:barrels?|tonnes?|tons?)\s+of\s+oil\s+equivalent)\b", re.I
)
_COMPANY_ONLY = re.compile(
    r"\b(?:China\s+)?National\s+(?:(?:Offshore|Iranian)\s+)?(?:Petroleum|Oil)\s+"
    r"(?:Corporation|Company)\b|\bJapan\s+Oil,\s*Gas\s+and\s+Metals\s+National\s+Corporation\b", re.I
)
_NON_COMMODITY_FUEL_REFERENCE = re.compile(
    r"\b(?:gasoline|diesel)[-\s]+(?:(?:powered|fueled|fuelled)[-\s]+)?"
    r"(?:vehicles?|cars?|trucks?|engines?|equipment|machinery|generators?|batter(?:y|ies))\b", re.I
)
_OTHER_COMMODITY = re.compile(
    r"\b(?:natural\s+gas|electricity|electric\s+power|coal|copper|iron\s+ore|"
    r"steel|gold|silver|uranium|alumin(?:um|ium)|zinc|metals?|wheat|maize)\b", re.I
)
_RESERVE_NAME = re.compile(r"\b(?:SPR|Strategic\s+Petroleum\s+Reserve)\b", re.I)
_RESERVE_MARKET_CONTEXT = re.compile(
    r"\b(?:stocks?|stockpiles?|inventor(?:y|ies)|barrels?|bbls?|MMbbl|"
    r"caverns?|drawdowns?|releas(?:e|es|ed)|sales?|refiner\w*|"
    r"pipeline\s+(?:connections?|capacity)|emergency\s+distribution)\b", re.I
)
_ADMIN_IT = re.compile(
    r"\b(?:Microsoft\s+Office|Office\s+365|cloud\s+(?:email|computing|services)|"
    r"email|e-mail|cyber\s*security|CASB|Cloud\s+Access\s+Security\s+Broker|"
    r"phishing|data\s+loss\s+prevention)\b", re.I
)
_EDGE_TRAILING = re.compile(r"\b(?:and|or|but|of|to|from|with|for|by|at|into|than|the|a|an)\s*[.!?][\"'”’)]*\s*$", re.I)
_TERMINAL = re.compile(r"[.!?][\"'”’)]*\s*$")
_REFERENCE_HEADING = re.compile(r"^\s*(?:References?|Bibliography|Endnotes?)\b", re.I)
_FURNITURE = re.compile(
    r"(?:\|\s*Page\s*\d+|\bNext\s+release\s+date\s*:|\bTel(?:ephone)?\s*:|"
    r"\bSubscribe\s+(?:to|for)\b|\bPrivacy\s+(?:policy|notice)\b|"
    r"\bAll\s+rights\s+reserved\b|\bCookie\s+(?:policy|settings)\b|"
    r"\bSign\s+up\s+for\b|\bU\.S\.\s+energy\s+market\s+indicators\b)", re.I
)
_PERIOD = re.compile(
    r"\b(?:18|19|20)\d{2}(?:[QH][1-4])?\b|\b[1-4][QH]\d{2}\b|"
    r"\b(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b|"
    r"\b(?:Quarter|Q)\s*[1-4]\b|"
    r"\b(?:January|February|March|April|June|July|August|September|October|November|December|"
    r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\b|"
    r"\bMay\s+(?:\d{1,4}|of\s+\d{4})\b|\b(?:in|during|for|since|by)\s+May\b|"
    r"\b(?:today|yesterday|currently|current|last\s+(?:week|month|year|quarter)|"
    r"this\s+(?:week|month|year|quarter)|recent\s+years)\b", re.I
)
_TEMPORAL_HEADING = re.compile(
    r"^(?:The\s+)?(?:main\s+points|key\s+(?:points|statistics|results)|"
    r"(?:oil\s+|energy\s+)?(?:statistics|results|production\s+summary))\s+"
    r"(?:for|in|during)\s+[^.!?\n]{1,100}:$", re.I
)
_PERCENT_CHANGE = re.compile(
    r"\b(?:rose|rise[ns]?|grew|grown|grow[ns]?|fell|fall[ns]?|"
    r"increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|"
    r"declin(?:e|es|ed|ing)|dropp?(?:ed|ing)|drop(?:s)?|lower|higher|more|less)\b", re.I
)
_STATISTIC_SUBJECT = re.compile(
    r"\b(?:production|output|imports?|exports?|demand|consumption|prices?|"
    r"inventor(?:y|ies)|stocks?|capacity|capability|deliver(?:y|ies)|quantit(?:y|ies))\b", re.I
)
_COMPARISON_ONLY = re.compile(
    r"\b(?:compared\s+(?:with|to)|relative\s+to|versus|vs\.|"
    r"(?:lower|higher|more|less)\s+than)\s+[^,;.!?]*", re.I
)
_SURVEY_FRACTION = re.compile(
    r"^(?:\d+(?:\.\d+)?|[A-Za-z]+(?:[-\s][A-Za-z]+)?)\s*(?:%|(?:percent|per\s+cent)\b)"
    r"[^.!?]*\b(?:expect(?:s|ed)?|anticipat(?:e|es|ed)|believ(?:e|es|ed))\b", re.I
)
_SURVEY_TARGET = re.compile(
    r"\b(?:production|output|imports?|exports?|demand|consumption|prices?|"
    r"completion(?:s)?|DUCs?|rigs?|rig\s+count|wells?|drilling|employment|"
    r"employees?|head\s*counts?|capital\s+(?:spending|expenditures?)|"
    r"investments?|revenues?|margins?|profits?|costs?)\b", re.I
)
_SURVEY_DENOMINATOR = re.compile(r"\b(?:executives|respondents|firms|participants|companies|businesses)\b", re.I)
_COHORT_ANAPHOR = re.compile(r"\b(?:these|those|such)\s+(?:projects|refineries|facilities|expansions|estimates)\b", re.I)
_AGGREGATE = re.compile(r"\b(?:account\s+for|combined|collectively|together|in\s+aggregate|total(?:s|led)?|add)\b", re.I)
_COHORT_ANCHOR = re.compile(
    r"\b(?:\d+|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:(?:new|refinery|refining|oil|planned)\s+)*(?:projects|refineries|facilities|expansions)\b", re.I
)


def oil_scope_supported(text: str) -> bool:
    """Mineral-oil evidence; generic sanctions/tankers are not sole anchors."""
    candidate = _NON_COMMODITY_FUEL_REFERENCE.sub("", _COMPANY_ONLY.sub("", _NON_MINERAL_OIL.sub("", text)))
    candidate = _RESERVE_NAME.sub("", candidate)
    return bool(_OIL.search(candidate) or (
        _RESERVE_NAME.search(text) and _RESERVE_MARKET_CONTEXT.search(text)
        and not _ADMIN_IT.search(text)))


def _visible_start(text: str) -> str:
    return re.sub(r"^[\s•▪◦\-–—\"'“‘(]+", "", text)


def _looks_dependent(text: str) -> bool:
    start = _visible_start(text)
    target_clause = re.split(r'\bif\b|\bat\s+(?:below|above|under|over)\b', text, maxsplit=1, flags=re.I)[0]
    survey_context_missing = (_SURVEY_FRACTION.match(start) and _PERCENT_CHANGE.search(text)
                              and (not _SURVEY_TARGET.search(target_clause) or not _SURVEY_DENOMINATOR.search(text)))
    return bool(survey_context_missing or _DEPENDENT.match(start) or _INTERNAL_ANAPHOR.search(text)
                or (_REGIONAL_PRICE_CONTINUATION.match(start) and not oil_scope_supported(text))
                or (_CONTEXTUAL_SUBJECT.match(start) and not oil_scope_supported(text)))


def _temporal_heading(text: str) -> bool:
    return bool(_TEMPORAL_HEADING.fullmatch(_visible_start(text).strip()) and _PERIOD.search(text))


def _statistical_period_missing(piece: str, page_has_period: bool) -> bool:
    """A percentage change is not dated by a comparison year or publication.

    This is intentionally narrow: static ratios, theoretical/model parameters,
    conditional mechanisms and explicit today/month/quarter observations are not
    rejected merely for omitting a four-digit year.
    """
    if not page_has_period or not _STATISTIC_SUBJECT.search(piece):
        return False
    if not (re.search(r"\d", piece) and re.search(r"%|\b(?:percent|per\s+cent)\b", piece, re.I)
            and _PERCENT_CHANGE.search(piece)):
        return False
    if re.search(r"\b(?:assum(?:e|es|ed|ption)|model|hypothetical|theoretical|scenario)\b", piece, re.I):
        return False
    if re.match(r"^(?:If|When|Whenever)\b", _visible_start(piece), re.I):
        return False
    current_period_text = _COMPARISON_ONLY.sub("", piece)
    return not bool(_PERIOD.search(current_period_text))


def _is_sentence_dot(text: str, at: int, after: int, sentence_start: int) -> bool:
    if text[at] != ".":
        return True
    if at and at + 1 < len(text) and text[at - 1].isdigit() and text[at + 1].isdigit():
        return False
    prefix = text[max(0, at - 36):at + 1]
    # Decimal dots are dealt with above; a country abbreviation may terminate a
    # full sentence if the next sentence supplies an unambiguous opening cue.
    if _COUNTRY_ABBREVIATION.search(prefix):
        following = _visible_start(text[after:])
        return bool(_FINITE_VERB.search(text[sentence_start:at + 1]) and _FINAL_BOUNDARY_CUE.match(following))
    if _ABBREVIATION.search(prefix):
        return False
    # A single initial is not a full stop: "J. Smith" / "A. OPEC ...".
    if re.search(r"\b[A-Z]\.$", prefix):
        return False
    return True


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return contiguous, trimmed exact intervals; no punctuation is removed.

    Blank paragraph breaks can isolate an unpunctuated fragment.  Single newlines
    are treated as PDF line wrapping, not guessed sentence boundaries.
    """
    spans: list[tuple[int, int]] = []
    start, cursor, size = 0, 0, len(text)
    while cursor < size:
        if cursor == start:
            while start < size and text[start].isspace():
                start += 1
            cursor = start
            if cursor == size:
                break
        if text[cursor] in ".!?":
            end = cursor + 1
            while end < size and text[end] in "\"'”’)]":
                end += 1
            if end == size or text[end].isspace():
                after = end
                while after < size and text[after].isspace():
                    after += 1
                if end == size or _is_sentence_dot(text, cursor, after, start):
                    spans.append((start, end))
                    start = cursor = end
                    continue
        if text[cursor] in "\r\n":
            paragraph = re.match(r"(?:\r?\n)[ \t]*(?:\r?\n)", text[cursor:])
            if paragraph:
                end = cursor
                while end > start and text[end - 1].isspace():
                    end -= 1
                if end > start:
                    spans.append((start, end))
                start = cursor = cursor + paragraph.end()
                continue
        cursor += 1
    end = size
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        spans.append((start, end))
    return spans


def _quality_reason(text: str, policy: SegmentationPolicy, *, initial_edge: bool = False) -> str | None:
    visible = _visible_start(text)
    count = len(words(text))
    if not visible or not re.match(r"[A-Z0-9]", visible):
        return "incomplete_initial_edge"
    # A page may begin with the state/country suffix of a named place from the
    # previous page. Never promote that clipped coordinate list to a short fact.
    # This is an edge check, not a ban on abbreviations inside intact sentences.
    if initial_edge and re.match(r'^[A-Z]{2,3},\s+(?:and|or)\b', visible):
        return "clipped_coordinate_subject_at_initial_edge"
    if not _TERMINAL.search(text):
        return "incomplete_final_edge"
    if _EDGE_TRAILING.search(text):
        return "incomplete_trailing_clause"
    if re.search(r"\b(?:and|or|of|for|to|with|from|in|by)\s*[•▪◦]", text, re.I):
        return "interrupted_clause"
    if re.search(r"(?:\b[A-Z]\s+){8,}", text):
        return "spaced_letter_furniture"
    if _FURNITURE.search(text) or re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", text, re.I):
        return "report_or_contact_furniture"
    if re.search(r"\(\d{3}\)\s*\d{3}[-\s]\d{4}\b", text):
        return "telephone_contact_furniture"
    if re.search(r"\b(?:However|But|Nevertheless),?\s+[A-Z][A-Z0-9/-]{1,12}\s+\d{1,3}\s+(?:The|This|A|An)\b", text):
        return "footnote_interrupted_clause"
    if not _FINITE_VERB.search(text) and re.search(
            r",\s*(?:keeping|leaving|leading|resulting|reflecting|causing|making)\b", text, re.I):
        return "finite_predicate_missing_in_clipped_clause"
    if _REFERENCE_HEADING.search(visible):
        return "reference_only"
    quoted_titles = re.findall(r'[“\"][^”\"]{8,}[”\"]', text)
    unquoted = re.sub(r'[“\"][^”\"]*[”\"]', '', text)
    if re.match(r'^\d{1,3}\s+\S', visible) and quoted_titles and not _FINITE_VERB.search(unquoted):
        return "numbered_quoted_title_reference"
    citation_years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?(?:[,).]|-)", text))
    titles = len(re.findall(r'[“\"][^”\"]{8,}[”\"]', text))
    if citation_years >= 3 and (text.count("http") >= 2 or titles >= 2):
        return "reference_heavy"
    if re.search(r'\b(?:is|are)\s+(?:of\s+)?the\s+(?:form|following\s+form)\s*:\s*(?:Where|In\s+which)\b', text, re.I):
        return "missing_formula_before_variable_definitions"
    if re.match(r'^(?:(?:Petroleum|Oil)\s+)?flow\s+chart\b', visible, re.I) and re.search(r'\bNotes?\s*:', text, re.I):
        return "diagram_caption_without_evidence"
    if len(re.findall(r'\b[A-D]\.\s*(?:19|20)\d{2}\s*:', text)) >= 2:
        return "reordered_figure_panel_furniture"
    if re.search(r'\bTotal\s+Percentage\s+of\s+GOM\b|\bOil,\s*BOPD\s+Shut-in\b|\bGas,\s*MMCFD\s+Shut-in\b', text, re.I):
        return "flattened_production_table"
    if re.search(r"\b(?:Figure|Table)\s+[A-Z0-9]", text, re.I) and not _FINITE_VERB.search(text):
        return "table_or_figure_fragment"
    tokens = text.split()
    numeric = sum(bool(re.fullmatch(r"[-+$£€\d.,()%/]+", token)) for token in tokens)
    if len(tokens) >= 8 and numeric / len(tokens) > .30:
        return "numeric_table_fragment"
    if count > policy.maximum_words:
        return "single_sentence_above_maximum"
    # An abbreviation at the end can be a clipped subject rather than a full
    # sentence.  Only country names with a preceding explicit finite clause get
    # the ordinary terminal interpretation; units/names alone are insufficient.
    tail = text[-36:]
    if _ABBREVIATION.search(tail) or _COUNTRY_ABBREVIATION.search(tail):
        if not (_COUNTRY_ABBREVIATION.search(tail) and _FINITE_VERB.search(text)):
            return "uncertain_terminal_abbreviation"
    if policy.require_oil_scope and _OTHER_COMMODITY.search(text) and not oil_scope_supported(text):
        return "out_of_scope_topic_switch"
    if policy.require_oil_scope and _ADMIN_IT.search(text) and not oil_scope_supported(text):
        return "administrative_it_without_oil_evidence"
    return None


def _record(text: str, start: int, end: int, **extra) -> dict:
    piece = text[start:end]
    return dict(text=piece, start=start, end=end, word_count=len(words(piece)), **extra)


def _chunk_issue(piece: str, policy: SegmentationPolicy, page_has_period: bool = False,
                 page_has_forecast: bool = False) -> str | None:
    pieces = [piece[a:b] for a, b in sentence_spans(piece)] if '?' in piece else []
    questions = [part for part in pieces if _pure_question(part)]
    if questions and not any(not _pure_question(part) and _FINITE_VERB.search(part)
                             and _TERMINAL.search(part) for part in pieces):
        return "pure_question_without_factual_answer"
    if policy.require_oil_scope and not oil_scope_supported(piece):
        return "no_mineral_oil_evidence"
    if _statistical_period_missing(piece, page_has_period):
        return "statistical_period_context_missing"
    if (page_has_forecast and page_has_period and len(words(piece)) < policy.minimum_words
            and re.search(r"\bemissions?\b", piece, re.I)
            and re.search(r"\b(?:grow|grows|rise|rises|reductions?|decreas(?:e|es)|declin(?:e|es))\b", piece, re.I)
            and not _PERIOD.search(piece)):
        return "forecast_emissions_period_context_missing"
    if len(words(piece)) < policy.minimum_words:
        # The minimum is a soft packing preference, not a requirement to invent
        # padding.  Short named-subject factual statements can stand alone.
        if len(words(piece)) < 7 or not _FINITE_VERB.search(piece):
            return "low_information_short_fragment"
    return None


def _partition(text: str, bundles: list[tuple[int, int]], policy: SegmentationPolicy) -> list[dict]:
    """Minimise missing evidence first, then balanced length and short tails.

    A dependent bundle is never cut; the soft maximum may be exceeded only for
    that one bundle.  Unrelated material cannot be rescued with arbitrary long
    chunks.  Dynamic programming avoids greedy last-chunk fragments.
    """
    size = len(bundles)
    if not size:
        return []
    counts = [len(words(text[start:end])) for start, end in bundles]
    totals = [0]
    for value in counts:
        totals.append(totals[-1] + value)
    costs: list[tuple[int, float] | None] = [None] * (size + 1)
    previous: list[tuple[int, str | None] | None] = [None] * (size + 1)
    costs[0] = (0, 0.0)
    page_has_period = bool(_PERIOD.search(text))
    page_has_forecast = bool(re.search(r"\bforecast(?:s|ed)?\b", text, re.I))
    for end in range(1, size + 1):
        for start in range(end - 1, -1, -1):
            count = totals[end] - totals[start]
            if count > policy.maximum_words and end - start > 1:
                break
            if count > policy.maximum_context_words:
                continue
            base = costs[start]
            if base is None:
                continue
            piece = text[bundles[start][0]:bundles[end - 1][1]]
            issue = _chunk_issue(piece, policy, page_has_period, page_has_forecast)
            missing = count if issue else 0
            # Lower-than-minimum factual chunks are allowed, but a neighboring
            # complete context within the ordinary maximum is preferred.
            length_cost = ((count - policy.target_words) / policy.target_words) ** 2
            if count < policy.minimum_words:
                length_cost += 1.0
            proposal = (base[0] + missing, base[1] + length_cost + .15)
            if costs[end] is None or proposal < costs[end]:
                costs[end] = proposal
                previous[end] = (start, issue)
    output: list[dict] = []
    end = size
    while end:
        choice = previous[end]
        if choice is None:
            raise AssertionError("Unpartitionable bounded bundle")
        start, issue = choice
        exceptions = []
        count = totals[end] - totals[start]
        if count < policy.minimum_words and not issue:
            exceptions.append("standalone_short_fact")
        if count > policy.maximum_words:
            exceptions.append("dependent_context_above_soft_maximum")
        if _temporal_heading(re.split(r"\r?\n\s*\r?\n", text[bundles[start][0]:bundles[start][1]], maxsplit=1)[0]):
            exceptions.append("temporal_heading_context")
        output.append(_record(text, bundles[start][0], bundles[end - 1][1],
                              reason=issue, exceptions=exceptions))
        end = start
    output.reverse()
    return output


def _accounting(text: str, accepted: list[dict], quarantined: list[dict]) -> dict:
    intervals = sorted(accepted + quarantined, key=lambda record: record["start"])
    cursor = 0
    characters = 0
    for record in intervals:
        start, end = record["start"], record["end"]
        if not (cursor <= start < end <= len(text)) or text[start:end] != record["text"]:
            raise AssertionError("Non-contiguous or altered segmentation provenance")
        if text[cursor:start].strip():
            raise AssertionError("Unaccounted non-whitespace text")
        characters += sum(not char.isspace() for char in text[start:end])
        cursor = end
    if text[cursor:].strip():
        raise AssertionError("Unaccounted final non-whitespace text")
    input_words = len(words(text))
    accepted_words = sum(record["word_count"] for record in accepted)
    quarantined_words = sum(record["word_count"] for record in quarantined)
    if input_words != accepted_words + quarantined_words:
        raise AssertionError("Segmentation changed word accounting")
    input_characters = sum(not char.isspace() for char in text)
    if input_characters != characters:
        raise AssertionError("Segmentation changed character accounting")
    return dict(input_words=input_words, accepted_words=accepted_words,
                quarantined_words=quarantined_words,
                non_whitespace_characters=input_characters,
                accounted_non_whitespace_characters=characters)


def _pure_question(text: str) -> bool:
    return bool(re.search(r"\?[\"'”’)]*\s*$", text))


def _unqualified_aggregate_tails(text):
    """Hold deictic totals without a complete explicit local project cohort.

    Do not attach a regional subset to a wider 'these projects' total. Earlier
    facility facts remain independently usable; no missing members are guessed
    or copied from non-contiguous source text.
    """
    tails = []
    for match in re.finditer(r'\S(?:.*?\S)?(?=(?:\r?\n\s*\r?\n)|\Z)', text, re.S):
        block = match.group()
        if (_COHORT_ANAPHOR.search(block) and _AGGREGATE.search(block) and re.search(r'\d', block)
                and not _COHORT_ANCHOR.search(block)):
            first = next((a for a, b in sentence_spans(block) if _COHORT_ANAPHOR.search(block[a:b])), None)
            if first is not None:
                tails.append((match.start() + first, match.end()))
    return tails


def segment_page(text: str, policy: SegmentationPolicy | dict | None = None,
                 *, hard_boundaries: list[int] | tuple[int, ...] | None = None,
                 required_contexts: list[dict] | tuple[dict, ...] | None = None) -> dict:
    """Apply the same semantic rules to *every* cleaned source page.

    Inputs and returned offsets use Python character indices, not UTF-8 bytes.
    ``exceptions`` describes supported deviations from soft size preferences;
    it never waives a sentence, topicality or provenance quality check.
    """
    if not isinstance(text, str):
        raise TypeError("Page text must be a string")
    if policy is None:
        policy = DEFAULT_POLICY
    elif isinstance(policy, dict):
        policy = SegmentationPolicy(**policy)
    if not isinstance(policy, SegmentationPolicy):
        raise TypeError("policy must be SegmentationPolicy or its field dictionary")
    if required_contexts is not None and not isinstance(required_contexts, (list, tuple)):
        raise ValueError('Required source contexts must be an ordered list of exact intervals')
    contexts = list(required_contexts or [])
    last_end = 0
    for context in contexts:
        if (not isinstance(context, dict) or type(context.get('start')) is not int
                or type(context.get('end')) is not int or not last_end <= context['start'] < context['end'] <= len(text)
                or not isinstance(context.get('question'), str) or not context['question'].strip()
                or '?' not in context['question']):
            raise ValueError('Required source contexts must be ordered nonoverlapping exact question intervals')
        last_end = context['end']
    if hard_boundaries is not None:
        if not isinstance(hard_boundaries, (list, tuple)) or any(
                type(value) is not int or not 0 < value < len(text) for value in hard_boundaries):
            raise ValueError("Source hard boundaries must be interior character offsets")
        if list(hard_boundaries) != sorted(set(hard_boundaries)):
            raise ValueError("Source hard boundaries must be unique and ordered")
    if contexts:
        boundaries = list(hard_boundaries or [])
        if any(context['start'] < edge < context['end'] for context in contexts for edge in boundaries):
            raise ValueError('Required source question context crosses an independent source boundary')
        edges = sorted(set([0, len(text), *boundaries, *[edge for context in contexts for edge in (context['start'], context['end'])]]))
        required = {(context['start'], context['end']): context for context in contexts}
        accepted, quarantined = [], []
        recoverable = {'no_mineral_oil_evidence', 'pure_question_without_factual_answer',
                       'statistical_period_context_missing', 'dependent_opening_without_local_context',
                       'low_information_short_fragment'}
        for a, b in zip(edges, edges[1:]):
            local = segment_page(text[a:b], policy)
            context = required.get((a, b))
            if context:
                start = a + len(text[a:b]) - len(text[a:b].lstrip())
                end = b - len(text[a:b]) + len(text[a:b].rstrip())
                piece = text[start:end]
                reason = ('source_question_temporal_context_missing' if context.get('question_is_local') is False or context['question'] not in piece else
                          'source_question_context_above_safety_maximum' if len(words(piece)) > policy.maximum_context_words else None)
                fatal = next((row['reason'] for row in local['quarantined'] if row['reason'] not in recoverable), None)
                reason = reason or fatal or _chunk_issue(piece, policy, bool(_PERIOD.search(text)),
                                                         bool(re.search(r'\bforecast(?:s|ed)?\b', text, re.I)))
                if reason:
                    quarantined.append(_record(text, start, end, reason=reason))
                else:
                    exceptions = ['source_native_question_context']
                    if len(words(piece)) > policy.maximum_words:
                        exceptions.append('dependent_context_above_soft_maximum')
                    if len(words(piece)) < policy.minimum_words:
                        exceptions.append('standalone_short_fact')
                    accepted.append(_record(text, start, end, exceptions=exceptions))
                continue
            for name, target in [('accepted', accepted), ('quarantined', quarantined)]:
                for row in local[name]:
                    shifted = dict(row, start=row['start'] + a, end=row['end'] + a)
                    issue = (_chunk_issue(row['text'], policy, bool(_PERIOD.search(text)),
                                         bool(re.search(r'\bforecast(?:s|ed)?\b', text, re.I))) if name == 'accepted' else None)
                    if issue:
                        shifted.pop('exceptions', None)
                        quarantined.append(dict(shifted, reason=issue))
                    else:
                        target.append(shifted)
        accepted.sort(key=lambda row: row['start'])
        quarantined.sort(key=lambda row: row['start'])
        return dict(accepted=accepted, quarantined=quarantined, policy=local['policy'],
                    accounting=_accounting(text, accepted, quarantined))
    if hard_boundaries is not None:
        if hard_boundaries:
            accepted, quarantined = [], []
            edges = [0, *hard_boundaries, len(text)]
            for a, b in zip(edges, edges[1:]):
                local = segment_page(text[a:b], policy)
                for name, target in [('accepted', accepted), ('quarantined', quarantined)]:
                    for row in local[name]:
                        shifted = dict(row, start=row['start'] + a, end=row['end'] + a)
                        # Boundaries reset evidence ownership, not page-wide
                        # quality flags. A dated page cannot make an undated
                        # percentage change valid merely by isolating a speaker.
                        issue = (_chunk_issue(row['text'], policy, bool(_PERIOD.search(text)),
                                             bool(re.search(r'\bforecast(?:s|ed)?\b', text, re.I))) if name == 'accepted' else None)
                        if issue:
                            shifted.pop('exceptions', None)
                            quarantined.append(dict(shifted, reason=issue))
                        else:
                            target.append(shifted)
            accepted.sort(key=lambda row: row['start'])
            quarantined.sort(key=lambda row: row['start'])
            return dict(accepted=accepted, quarantined=quarantined, policy=local['policy'],
                        accounting=_accounting(text, accepted, quarantined))
    accepted: list[dict] = []
    quarantined: list[dict] = []
    run: list[tuple[int, int]] = []
    previous_good = False
    previous_end: int | None = None
    pending_temporal_heading = False
    pending_question = False
    aggregate_tails = _unqualified_aggregate_tails(text)

    def flush():
        nonlocal run, pending_temporal_heading, pending_question
        for record in _partition(text, run, policy):
            reason = record.pop("reason")
            if reason:
                record.pop("exceptions")
                quarantined.append(dict(record, reason=reason))
            else:
                accepted.append(record)
        run = []
        pending_temporal_heading = False
        pending_question = False

    for start, end in sentence_spans(text):
        paragraph_break = previous_end is not None and bool(re.search(r'\r?\n\s*\r?\n', text[previous_end:start]))
        previous_end = end
        piece = text[start:end]
        tail = next(((a, b) for a, b in aggregate_tails if a <= start < b), None)
        if tail:
            if start == tail[0]:
                flush()
                quarantined.append(_record(text, *tail, reason='aggregate_project_cohort_context_missing'))
            previous_good = False
            continue
        heading = _temporal_heading(piece)
        if heading:
            flush()
            run.append((start, end))
            previous_good = True
            pending_temporal_heading = True
            continue
        reason = _quality_reason(piece, policy, initial_edge=not text[:start].strip())
        question = _pure_question(piece)
        if question and reason is None:
            if not pending_temporal_heading and not pending_question:
                flush()
                run.append((start, end))
            else:
                run[-1] = (run[-1][0], end)
            if len(words(text[run[-1][0]:end])) > policy.maximum_context_words:
                overlong = run.pop()
                flush()
                quarantined.append(_record(text, *overlong, reason="question_context_above_safety_maximum"))
                previous_good = False
                continue
            previous_good = True
            pending_question = True
            continue
        # A question is context, not a factual short passage; only an adjacent
        # same-paragraph answer may bind. Explicit source boundaries above also
        # prevent borrowing an answer from a different respondent.
        if pending_question and paragraph_break:
            flush()
            previous_good = False
        dependent = _looks_dependent(piece) or pending_temporal_heading or pending_question
        # A valid time label may need the complete preceding non-oil summary
        # sentence to reach the first oil fact without non-contiguous copying.
        if pending_temporal_heading and reason == "out_of_scope_topic_switch":
            reason = None
        if reason is None and dependent and not previous_good:
            reason = "dependent_opening_without_local_context"
        if reason:
            flush()
            quarantined.append(_record(text, start, end, reason=reason))
            previous_good = False
            continue
        if paragraph_break and not dependent:
            flush()
        if dependent:
            if not run:
                raise AssertionError("Dependent sentence lost its preceding context")
            run[-1] = (run[-1][0], end)
            if len(words(text[run[-1][0]:end])) > policy.maximum_context_words:
                overlong = run.pop()
                flush()
                quarantined.append(_record(text, *overlong, reason="dependent_context_above_safety_maximum"))
                previous_good = False
                continue
            if pending_temporal_heading and oil_scope_supported(text[run[-1][0]:end]):
                pending_temporal_heading = False
            pending_question = False
        else:
            run.append((start, end))
        previous_good = True
    flush()
    accepted.sort(key=lambda record: record["start"])
    quarantined.sort(key=lambda record: record["start"])
    policy_data = asdict(policy)
    policy_data.update(overlap_words=0, sentence_boundary_only=True,
                       single_sentence_above_maximum="quarantine",
                       minimum_words_is_soft=True, maximum_words_is_soft=True,
                       source_question_temporal_context='bind_exact_contiguous_native_question_or_quarantine',
                       coordinate_initial_edge='quarantine_suffix_continuation',
                       temporal_heading_binding=True, unresolved_statistical_period="quarantine",
                       pure_question_requires_factual_answer=True,
                       survey_fraction_requires_local_target_and_cohort=True,
                       independent_source_boundaries="honor_source_native_offsets")
    return dict(accepted=accepted, quarantined=quarantined, policy=policy_data,
                accounting=_accounting(text, accepted, quarantined))
