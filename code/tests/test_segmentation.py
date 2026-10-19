"""Source-page semantic segmentation, not resegmentation of old passage IDs."""
from __future__ import annotations

import pytest

from oil_evidence.segmentation import SegmentationPolicy, oil_scope_supported, segment_page, sentence_spans
from oil_evidence.store import words


def sentence(index=0, count=35, opening=None):
    opening = opening or f"Crude oil production in region {index}"
    prefix = f"{opening} was reported at 13.7 million barrels per day in 2025"
    padding = " ".join(["according to the published official estimates"] * count)
    return f"{prefix} {padding}."


def assert_accounted(text, result):
    combined = sorted(result["accepted"] + result["quarantined"], key=lambda record: record["start"])
    cursor = 0
    for record in combined:
        assert record["start"] >= cursor
        assert not text[cursor:record["start"]].strip()
        assert text[record["start"]:record["end"]] == record["text"]
        assert record["word_count"] == len(words(record["text"]))
        cursor = record["end"]
    assert not text[cursor:].strip()
    accounting = result["accounting"]
    assert accounting["input_words"] == accounting["accepted_words"] + accounting["quarantined_words"]
    assert accounting["non_whitespace_characters"] == accounting["accounted_non_whitespace_characters"]


def test_default_policy_is_one_fixed_page_policy():
    result = segment_page("Crude oil output was 13.7 million barrels per day in 2025.")
    assert result["policy"]["version"] == "semantic-page-v1"
    assert result["policy"]["target_words"] == 160
    assert result["policy"]["minimum_words"] == 65
    assert result["policy"]["maximum_words"] == 240
    assert result["policy"]["maximum_context_words"] == 480
    assert result["policy"]["overlap_words"] == 0
    assert result["accepted"][0]["exceptions"] == ["standalone_short_fact"]
    assert not result["quarantined"]


@pytest.mark.parametrize("text", ["", " \r\n\t ", "Oil prices", "lowercase oil production was rising.", "Crude oil output rose to."])
def test_empty_and_fragments_are_fully_accounted(text):
    result = segment_page(text)
    assert not result["accepted"]
    assert_accounted(text, result)


def test_offsets_whitespace_and_unicode_remain_exact():
    text = ' \n“Crude oil output was 13.7 million barrels per day in 2025.”\n\n  It increased by 2% in 2026.\t'
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["text"] == text[result["accepted"][0]["start"]:result["accepted"][0]["end"]]
    assert "\n\n  " in result["accepted"][0]["text"]
    assert_accounted(text, result)


def test_no_hard_word_cut_of_single_long_sentence():
    text = sentence(count=50)
    assert len(words(text)) > 240
    result = segment_page(text)
    assert not result["accepted"]
    assert len(result["quarantined"]) == 1
    assert result["quarantined"][0]["text"] == text
    assert result["quarantined"][0]["reason"] == "single_sentence_above_maximum"


def test_many_independent_sentences_have_variable_lengths_and_no_overlap():
    text = " ".join(sentence(index=index, count=count) for index, count in enumerate([8, 10, 7, 9, 6, 11]))
    result = segment_page(text)
    assert len(result["accepted"]) >= 2
    assert not result["quarantined"]
    assert all(65 <= record["word_count"] <= 240 for record in result["accepted"])
    assert len({record["word_count"] for record in result["accepted"]}) > 1
    assert_accounted(text, result)


@pytest.mark.parametrize("opening", ["It", "This", "These", "They", "However", "Therefore", "Meanwhile", "In turn", "By contrast", "Yet it"])
def test_dependent_opening_is_never_a_new_chunk(opening):
    first = sentence(count=25)
    second = f"{opening} increased by 2% in 2026 according to the official forecast."
    third = sentence(index=1, count=20)
    text = first + " " + second + " " + third
    result = segment_page(text)
    assert not result["quarantined"]
    matching = [record for record in result["accepted"] if second in record["text"]]
    assert len(matching) == 1
    assert first in matching[0]["text"]
    assert_accounted(text, result)


@pytest.mark.parametrize("opening", ["It", "This", "These", "They", "However", "Therefore", "The forecast", "Production"])
def test_page_initial_dependency_fails_closed(opening):
    text = f"{opening} was expected to increase crude oil production by 2% in 2026."
    result = segment_page(text)
    # Contextual nouns are self-contained when their sentence explicitly names
    # the oil sector; demonstratives still need the missing antecedent.
    if opening in {"The forecast", "Production"}:
        assert result["accepted"]
    else:
        assert not result["accepted"]
        assert result["quarantined"][0]["reason"] == "dependent_opening_without_local_context"
    assert_accounted(text, result)


def test_dependency_after_fragment_does_not_attach_to_unrelated_good_evidence():
    text = "Crude oil production was 13 million barrels per day. lowercase extraction failed. It increased in 2026."
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert len(result["quarantined"]) == 2
    assert result["quarantined"][1]["reason"] == "dependent_opening_without_local_context"
    assert_accounted(text, result)


def test_bound_dependency_may_exceed_soft_maximum_but_is_not_cut():
    first = sentence(count=23)
    second = sentence(count=22, opening="It")
    assert len(words(first)) < 240 and len(words(second)) < 240
    text = first + " " + second
    assert 240 < len(words(text)) <= 480
    result = segment_page(text)
    assert not result["quarantined"]
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["text"] == text
    assert result["accepted"][0]["exceptions"] == ["dependent_context_above_soft_maximum"]


def test_dependency_above_safety_maximum_is_quarantined_without_cut():
    text = sentence(count=26) + " " + sentence(count=26, opening="It") + " " + sentence(count=26, opening="This")
    result = segment_page(text)
    assert not result["accepted"]
    assert result["quarantined"][0]["reason"] == "dependent_context_above_safety_maximum"
    assert result["quarantined"][0]["text"] == text
    assert_accounted(text, result)


def test_short_facts_join_neighbor_when_complete_context_fits():
    first = "Crude oil output was 13.7 million barrels per day in 2025."
    second = "Saudi Arabian crude oil production increased by 2% during 2026."
    result = segment_page(first + " " + second)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["text"] == first + " " + second


@pytest.mark.parametrize("fragment", ["Crude oil.", "Oil production 2025 2026 2027 13.7 13.8 14.3.", "Figure 3 Crude oil gasoline diesel distillate."])
def test_nominal_and_legend_fragments_do_not_count_as_short_facts(fragment):
    result = segment_page(fragment)
    assert not result["accepted"]
    assert result["quarantined"]
    assert_accounted(fragment, result)


@pytest.mark.parametrize("abbreviation", ["U.S.", "U.K.", "e.g.", "St.", "Inc.", "No.", "m.b."])
def test_abbreviations_inside_sentence_are_not_boundaries(abbreviation):
    text = f"Crude oil production at {abbreviation} Alpha was 13.7 million barrels per day in 2025."
    assert sentence_spans(text) == [(0, len(text))]
    result = segment_page(text)
    assert len(result["accepted"]) == 1


def test_decimal_currency_year_and_units_stay_literal():
    text = "U.S. crude oil output was 13.7 m.b. per day in 2025, and Brent was $69.25 per barrel in 2026."
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["text"] == text
    assert sentence_spans(text) == [(0, len(text))]


@pytest.mark.parametrize("text", [
    "U.S. Crude Oil Production was forecast at 13.7 million barrels per day in 2025.",
    "U.K. Petroleum exports were expected to increase by 2% in 2026.",
    "U. S. Crude oil output was expected to rise in 2026.",
])
def test_country_abbreviation_at_initial_subject_never_drops_country(text):
    assert sentence_spans(text) == [(0, len(text))]
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["text"] == text


def test_country_abbreviation_at_sentence_end_can_split_and_keep_dependency():
    first = "Saudi Arabia exported crude oil to the U.S."
    second = "It exported 1.2 million barrels per day in 2025."
    text = first + " " + second
    assert sentence_spans(text) == [(0, len(first)), (len(first) + 1, len(text))]
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert_accounted(text, result)


def test_terminal_unit_or_clipped_named_abbreviation_is_uncertain():
    text = "Crude oil output in St."
    result = segment_page(text)
    assert not result["accepted"]
    assert result["quarantined"][0]["reason"] == "uncertain_terminal_abbreviation"


@pytest.mark.parametrize("text", [
    "Treasury designated shipping companies and tankers under sanctions in 2025.",
    "Vegetable oil output was 13 million tonnes in 2025.",
    "Palm oil exports were forecast to increase in 2026.",
    "Natural gas reserves were 13 million barrels of oil equivalent in 2025.",
    "China National Offshore Oil Corporation was listed in Hong Kong in 2025.",
])
def test_generic_sanctions_tanker_vegetable_and_company_names_do_not_establish_scope(text):
    assert not oil_scope_supported(text)
    result = segment_page(text)
    assert not result["accepted"]
    assert_accounted(text, result)


def test_actual_crude_oil_sanctions_are_in_scope():
    text = "Treasury imposed sanctions on Iranian crude oil exports in 2025."
    result = segment_page(text)
    assert len(result["accepted"]) == 1


def test_lubricating_oil_is_petroleum_not_vegetable_oil():
    text = "Lubricating oil production was 1.2 million barrels per day in 2025."
    assert oil_scope_supported(text)
    assert len(segment_page(text)["accepted"]) == 1


def test_reference_heavy_and_furniture_are_quarantined():
    text = 'References 2022, "Crude oil supply report" http://a.test 2023, "Petroleum market outlook" http://b.test 2024.'
    result = segment_page(text)
    assert not result["accepted"]
    assert_accounted(text, result)


@pytest.mark.parametrize("text", [
    "Crude oil production was 13 million barrels per day. Next release date: 2026-11-01.",
    "Crude oil production rose in 2025 | Page 3.",
    "S T R A T E G I C P E T R O L E U M R E S E R V E oil.",
    "Crude oil production rose in 2025 and • supply increased.",
    "Crude oil data questions are answered by energy@example.test.",
])
def test_furniture_and_interrupted_clause_cannot_hide_in_passage(text):
    result = segment_page(text)
    assert result["quarantined"]
    assert all("Next release date" not in record["text"] for record in result["accepted"])
    assert_accounted(text, result)


def test_explicit_non_oil_topic_switch_not_rescued_by_neighboring_oil():
    text = "Crude oil output was 13.7 million barrels per day in 2025. Natural gas output rose by 2% in 2026."
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert "Natural gas" not in result["accepted"][0]["text"]
    assert result["quarantined"][0]["reason"] == "out_of_scope_topic_switch"


def test_contextual_non_oil_sentence_binds_to_oil_evidence_not_independently_dropped():
    text = "Saudi Arabian crude oil output rose by 2% in 2025. Production was expected to rise again in 2026."
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["text"] == text
    assert not result["quarantined"]


def test_unpunctuated_paragraph_heading_and_tail_remain_accounted():
    text = "Oil supply overview\n\nCrude oil output was 13.7 million barrels per day in 2025.\n\nSource heading without terminator"
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert len(result["quarantined"]) == 2
    assert_accounted(text, result)


def test_policy_can_disable_scope_for_general_pipeline_test():
    text = "Natural gas output was 1.2 billion cubic feet per day in 2025."
    result = segment_page(text, SegmentationPolicy(require_oil_scope=False))
    assert len(result["accepted"]) == 1


@pytest.mark.parametrize("values", [
    {"minimum_words": 0}, {"target_words": True}, {"maximum_words": 64},
    {"maximum_context_words": 200}, {"target_words": 241}, {"version": "../escape"},
    {"require_oil_scope": "yes"},
])
def test_invalid_policy_fails(values):
    with pytest.raises(ValueError):
        SegmentationPolicy(**values)


def test_dictionary_policy_and_type_errors():
    assert segment_page("", {"target_words": 170})["policy"]["target_words"] == 170
    with pytest.raises(TypeError):
        segment_page(None)
    with pytest.raises(TypeError):
        segment_page("", "not a policy")


def test_repeat_calls_are_deterministic():
    text = " ".join(sentence(index=index, count=8) for index in range(8))
    assert segment_page(text) == segment_page(text)


def test_real_legal_et_al_v_cir_citation_is_not_clipped():
    text = "The oil analysis follows the court's ruling in Center for Biological Diversity, et. al. v. Department of the Interior, 563 F.3d 466 (D.C. Cir. 2009), which ruled that consumption cannot be analyzed."
    assert sentence_spans(text) == [(0, len(text))]
    assert segment_page(text)["accepted"][0]["text"] == text


@pytest.mark.parametrize("opening", ["For instance", "For example", "In other words", "Similarly", "Since then", "Moving forward"])
def test_explanation_and_temporal_openings_require_their_antecedent(opening):
    first = "U.S. crude oil exports began in 2014."
    second = f"{opening}, crude oil exports increased by 2% in 2016."
    assert not segment_page(second)["accepted"]
    result = segment_page(first + " " + second)
    assert result["accepted"][0]["text"] == first + " " + second


@pytest.mark.parametrize("dependent", [
    "Oil and natural gas prices could experience volatility during this period.",
    "According to the agency, this increase was caused by lower crude oil prices.",
    "GDP growth remains a key driver of oil demand in these economies.",
    "In 2020, Malaysia was the largest destination for these exports of petroleum.",
    "Oil-fired capacity, the next largest total, stood at 8.1 GW.",
    "Most of the production gain is expected in Brazil, with crude oil output rising.",
])
def test_anaphor_inside_sentence_binds_even_when_oil_keyword_is_present(dependent):
    previous = "UK crude oil production was reported to rise during 2025."
    assert not segment_page(dependent)["accepted"]
    result = segment_page(previous + " " + dependent)
    assert result["accepted"][0]["text"] == previous + " " + dependent


def test_real_numbered_fitch_report_is_not_a_short_fact():
    text = '10 Fitch Solutions Group, “Indonesia Oil & Gas Report Q2 2025.”'
    result = segment_page(text)
    assert not result["accepted"]
    assert result["quarantined"][0]["reason"] == "numbered_quoted_title_reference"


def test_actual_numbered_reported_quote_with_a_finite_clause_is_not_bibliography():
    text = '1 Officials reported that crude oil exports rose by 2% in 2025, saying “production rose.”'
    assert segment_page(text)["accepted"]


def test_real_spr_it_only_paragraph_does_not_get_scope_from_organization_name():
    text = "The SPR uses Microsoft Office 365 to leverage cloud services for email and Microsoft Office capabilities. Cyber security controls were extended to the cloud email services."
    result = segment_page(text)
    assert not oil_scope_supported(text)
    assert not result["accepted"]
    assert all(record["reason"] == "administrative_it_without_oil_evidence" for record in result["quarantined"])


def test_reserve_stock_cavern_and_drawdown_facts_remain_valid():
    text = "The SPR held 638.1 MMbbl in 2020. The SPR operates six storage caverns for emergency drawdowns."
    assert oil_scope_supported(text)
    assert segment_page(text)["accepted"]


def test_petroleum_name_alone_does_not_rescue_spr_it_text():
    text = "The Strategic Petroleum Reserve uses Microsoft Office 365 for its cloud email services."
    assert not oil_scope_supported(text)
    assert not segment_page(text)["accepted"]


def test_real_oil_pipeline_cyber_event_is_not_administrative_it_only():
    text = "The cyber security attack interrupted crude oil flows through the pipeline in 2025."
    assert segment_page(text)["accepted"]


def test_real_flowchart_only_notes_are_not_oil_evidence():
    text = "Petroleum flow chart 2025 (million tonnes) Notes: This flow chart is based on the data in Tables 3.1 and 3.2. The numbers on either side will not match due to losses in transformation. Biofuels are not included."
    assert not segment_page(text)["accepted"]


def test_missing_diversity_formula_not_reconstructed_or_indexed():
    text = "The crude oil diversity index is of the form: Where x is the proportion of fuel supply represented by the source country."
    result = segment_page(text)
    assert not result["accepted"]
    assert result["quarantined"][0]["reason"] == "missing_formula_before_variable_definitions"


def test_explicit_formula_and_units_are_preserved_not_rejected_as_missing():
    text = "The crude oil self-sufficiency ratio is of the form: production (kt) divided by consumption (kt), where kt is thousand tonnes."
    assert segment_page(text)["accepted"][0]["text"] == text


def test_flat_bsee_production_table_is_not_prose():
    text = "Total Percentage of GOM Production Oil, BOPD Shut-in 1,305,558 (BOPD) Gas, MMCFD Shut-in 1,684.20 (MMCFD)."
    result = segment_page(text)
    assert not result["accepted"]
    assert result["quarantined"][0]["reason"] == "flattened_production_table"


def test_reordered_figure_panel_labels_are_rejected():
    text = "Crude oil futures reflect the recovery. B. 2009: Recovery from the financial crisis A. 2008: Financial crisis Sources: World Bank."
    result = segment_page(text)
    assert any(record["reason"] == "reordered_figure_panel_furniture" for record in result["quarantined"])
    assert all("2009:" not in record["text"] for record in result["accepted"])


def test_complete_independent_paragraphs_do_not_mix_macro_trade_and_oil():
    macro = "Total U.S. exports are forecast to grow by 5.5% in 2018."
    oil = "Brent crude oil prices were $77.39 per barrel in July 2018."
    result = segment_page(macro + "\n\n" + oil)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["text"] == oil
    assert result["quarantined"][0]["text"] == macro


def test_explicit_dependent_paragraph_still_keeps_its_context():
    text = "Crude oil production rose by 2% in 2025.\n\nHowever, the crude oil forecast for 2026 was unchanged."
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["text"] == text
    assert_accounted(text, result)


@pytest.mark.parametrize("breaks", ["\n\n", "\r\n\r\n", "\n  \n"])
def test_exact_temporal_heading_binds_without_inventing_punctuation(breaks):
    heading = "The main points for the second quarter of 2018:"
    general = "Total energy production was 2.1 per cent lower than in the second quarter of 2017."
    oil = "Oil production rose by 0.4 per cent when compared with the second quarter of 2017, with crude oil production up 0.9 per cent."
    text = heading + breaks + general + breaks + oil
    result = segment_page(text)
    assert len(result["accepted"]) == 1
    assert not result["quarantined"]
    assert result["accepted"][0]["text"] == text
    assert "temporal_heading_context" in result["accepted"][0]["exceptions"]
    assert heading + "." not in result["accepted"][0]["text"]
    assert_accounted(text, result)


def test_orphan_temporal_heading_never_counts_as_a_fact():
    text = "The main points for the second quarter of 2018:"
    result = segment_page(text)
    assert not result["accepted"]
    assert_accounted(text, result)


def test_unpunctuated_bad_preceding_statistical_context_cannot_be_reused():
    previous = "Jet fuel demand in Quarter 1 2025 was down on the"
    fact = "Crude oil production dropped by 13 per cent because of refinery closures."
    text = previous + "\n\n" + fact
    result = segment_page(text)
    assert not result["accepted"]
    assert any(record["reason"] == "statistical_period_context_missing" for record in result["quarantined"])
    assert_accounted(text, result)


def test_comparison_period_alone_is_not_the_current_statistical_period():
    heading = "The main points for the second quarter of 2018:"
    bad = "oil production begins mid-clause."
    fact = "Oil production rose by 0.4 per cent when compared with the second quarter of 2017."
    result = segment_page(heading + "\n\n" + bad + "\n\n" + fact)
    assert not result["accepted"]
    assert any(record["reason"] == "statistical_period_context_missing" for record in result["quarantined"])


@pytest.mark.parametrize("period", ["in May 2025", "in April", "today", "last week", "in Quarter 1 2025", "in 2Q22", "in 2025Q1"])
def test_named_periods_and_daily_observations_are_allowed(period):
    text = f"U.S. crude oil production increased by 2 per cent {period}."
    assert segment_page(text)["accepted"]


def test_current_period_after_leading_comparison_is_not_deleted():
    text = "Compared with 2017, crude oil production rose by 2 per cent in 2025."
    assert segment_page(text)["accepted"]


def test_modal_may_is_not_a_calendar_date():
    text = "The main points for the second quarter of 2025:\n\nA lowercase broken fragment\n\nCrude oil production may grow by 2 per cent."
    assert not segment_page(text)["accepted"]


def test_static_ratios_and_generic_physical_mechanisms_do_not_need_a_period():
    text = "Crude oil reinjection is used to improve oil recovery. The degas plant reduces volatile organic compounds by 97 percent."
    assert segment_page(text)["accepted"]


def test_model_assumptions_are_not_observed_quarterly_statistical_changes():
    text = "The crude oil model assumes production costs decline by 1 percent through technological improvement."
    assert segment_page(text)["accepted"]


def test_forecast_emissions_explanation_cannot_lose_its_year_and_country():
    first = "We forecast U.S. CO2 emissions will decrease in 2026."
    unrelated = "Natural gas consumption fell in 2026."
    fact = "Slight reductions in petroleum emissions are associated with lower consumption of motor gasoline."
    text = first + " " + unrelated + " " + fact
    result = segment_page(text)
    assert all(fact not in row["text"] for row in result["accepted"])
    assert any(row["reason"] == "forecast_emissions_period_context_missing" for row in result["quarantined"])
    assert_accounted(text, result)


def test_qualified_petroleum_emissions_forecast_is_retained():
    text = "We forecast U.S. petroleum emissions will decrease in 2026 because of lower gasoline consumption."
    assert segment_page(text)["accepted"]


def test_crosspage_comma_gerund_without_predicate_is_not_rescued():
    clipped = "U.S. crude oil production, keeping Brent-linked crude oil production comparatively elevated."
    valid = "EIA forecasts U.S. crude oil production will average 10.6 million barrels per day in March 2021."
    result = segment_page(clipped + " " + valid)
    assert result["quarantined"][0]["reason"] == "finite_predicate_missing_in_clipped_clause"
    assert result["accepted"][0]["text"] == valid


def test_actual_finite_clause_with_comma_gerund_is_not_rejected():
    text = "U.S. crude oil production rose in 2025, keeping refineries well supplied."
    assert segment_page(text)["accepted"]


def test_phone_catalogue_cannot_hide_inside_method_prose():
    text = "Official model representative James Preciado Petroleum Analysis (202) 586-8769 Documentation EIA Liquid Fuels Market Module, December 2016."
    result = segment_page(text)
    assert not result["accepted"]
    assert result["quarantined"][0]["reason"] == "telephone_contact_furniture"


def test_interrupted_acronym_actor_footnote_is_quarantined():
    text = "However, BOEM 3 The baseline oil forecast uses the 2020 Annual Energy Outlook."
    result = segment_page(text)
    assert not result["accepted"]
    assert result["quarantined"][0]["reason"] == "footnote_interrupted_clause"


@pytest.mark.parametrize("opening", ["Over the same period", "During the same period", "As such"])
def test_additional_temporal_and_conclusion_continuations_keep_context(opening):
    first = "The crude oil companies earned $65.9 billion in the first half of 2022."
    second = f"{opening}, they invested 59 percent of their crude oil revenues in dividends."
    assert not segment_page(second)["accepted"]
    result = segment_page(first + " " + second)
    assert result["accepted"][0]["text"] == first + " " + second


def test_regional_price_continuation_keeps_explicit_diesel_subject():
    first = "The U.S. diesel fuel price was $3.29 per gallon on June 14, 2021."
    second = "The East Coast and West Coast prices increased to $3.28 and $3.80 per gallon."
    result = segment_page(first + " " + second)
    assert len(result["accepted"]) == 1
    assert result["accepted"][0]["text"] == first + " " + second
    assert not segment_page(second)["accepted"]


@pytest.mark.parametrize("direction", ["lower", "higher", "more", "less"])
def test_than_comparator_year_is_not_current_statistical_period(direction):
    period = "The main points for the third quarter of 2019:\n\n"
    bad = "broken context ends mid-clause\n\n"
    fact = f"Total energy production was 1.8 per cent {direction} than in the third quarter of 2018, with changes in oil output."
    result = segment_page(period + bad + fact)
    assert not result["accepted"]
    assert any(row["reason"] == "statistical_period_context_missing" for row in result["quarantined"])
    assert_accounted(period + bad + fact, result)


@pytest.mark.parametrize("direction", ["lower", "higher", "more", "less"])
def test_than_comparison_retains_explicit_current_year(direction):
    text = f"Crude oil production in the third quarter of 2019 was 1.8 per cent {direction} than in the third quarter of 2018."
    result = segment_page(text)
    assert result["accepted"][0]["text"] == text


@pytest.mark.parametrize("subject", ["Deliveries", "Delivery quantity", "Delivered quantities"])
def test_delivery_statistics_require_valid_period(subject):
    first = "Crude oil output was reported in Q2 2018.\n\n"
    bad = "missing period context ends on the\n\n"
    fact = f"{subject} of jet fuel increased by 0.3 per cent."
    result = segment_page(first + bad + fact)
    assert all(fact not in row["text"] for row in result["accepted"])
    assert any(row["reason"] == "statistical_period_context_missing" for row in result["quarantined"])
    assert_accounted(first + bad + fact, result)


@pytest.mark.parametrize("subject", ["Deliveries", "Delivery quantity", "Delivered quantities"])
def test_delivery_statistics_with_current_period_are_retained(subject):
    text = f"{subject} of jet fuel increased by 0.3 per cent in Q2 2018."
    result = segment_page(text)
    assert result["accepted"][0]["text"] == text


def test_delivery_percentage_can_keep_exact_temporal_heading():
    text = "The main points for the second quarter of 2018:\n\nDeliveries of jet fuel increased by 0.3 per cent."
    result = segment_page(text)
    assert result["accepted"][0]["text"] == text
    assert "temporal_heading_context" in result["accepted"][0]["exceptions"]


def test_explicit_current_period_after_comparison_comma_is_retained():
    text = "Crude oil production was 1.8 per cent lower than in 2018, during the third quarter of 2019."
    assert segment_page(text)["accepted"][0]["text"] == text


@pytest.mark.parametrize('question', [
    'Is oil and gas private equity over?',
    'Will crude oil production increase in 2027?',
    'At what oil price would firms complete drilled but uncompleted wells?',
])
def test_pure_question_is_never_a_standalone_fact(question):
    result = segment_page(question)
    assert not result['accepted']
    assert result['quarantined'][0]['reason'] == 'pure_question_without_factual_answer'
    assert_accounted(question, result)


def test_question_binds_same_paragraph_real_answer_not_other_respondent():
    question = 'At what oil price would firms complete drilled but uncompleted wells?'
    answer = 'Thirty-six percent of executives expect more DUC completions if oil prices reach $46 per barrel.'
    local = question + '\n' + answer
    assert segment_page(local)['accepted'][0]['text'] == local
    separated = question + '\n\n' + answer
    result = segment_page(separated, hard_boundaries=[len(question) + 2])
    assert all(question not in row['text'] for row in result['accepted'])
    assert any(row['text'] == question for row in result['quarantined'])
    assert_accounted(separated, result)


@pytest.mark.parametrize('fraction', ['Two percent', '2%', 'Eighteen per cent'])
def test_survey_change_fraction_without_local_target_and_cohort_is_held(fraction):
    tail = fraction + ' expect a substantial increase at below $40 a barrel.'
    assert not segment_page(tail)['accepted']
    first = 'Thirty-six percent of executives expect a substantial increase in DUC completions if oil prices reach $46 per barrel.'
    result = segment_page(first + ' ' + tail)
    assert result['accepted'][0]['text'] == first + ' ' + tail
    assert_accounted(first + ' ' + tail, result)


def test_oil_price_in_if_condition_is_not_the_changed_survey_target():
    first = 'Thirty-six percent of executives expect more DUC completions if oil prices reach $46 per barrel.'
    second = 'Twenty-eight percent of executives expect a substantial increase if oil prices reach $51 per barrel.'
    assert not segment_page(second)['accepted']
    result = segment_page(first + ' ' + second)
    assert len(result['accepted']) == 1 and first in result['accepted'][0]['text']


def test_complete_survey_target_denominator_is_self_contained():
    text = 'Twenty percent of executives expect crude oil prices to increase in 2027.'
    assert segment_page(text)['accepted'][0]['text'] == text


def test_independent_native_boundaries_never_merge_three_respondents():
    comments = ['Crude oil prices are too low for our company.',
                'Our refinery will increase oil processing in 2027.',
                'The crude oil market remains volatile in 2027.']
    text = '\n\n'.join(comments)
    boundaries = [len(comments[0]) + 2, len(comments[0]) + len(comments[1]) + 4]
    result = segment_page(text, hard_boundaries=boundaries)
    assert [row['text'] for row in result['accepted']] == comments
    assert_accounted(text, result)


def test_hard_boundary_does_not_allow_pronoun_or_fraction_borrowing():
    first = 'Crude oil production at our company increased in 2027.'
    second = 'This was expected to increase by two percent in 2027.'
    text = first + '\n\n' + second
    result = segment_page(text, hard_boundaries=[len(first) + 2])
    assert result['accepted'][0]['text'] == first
    assert any(row['text'] == second for row in result['quarantined'])
    assert_accounted(text, result)


def test_hard_boundary_cannot_erase_page_statistical_period_guard():
    first = 'Crude oil output was reported in 2027.'
    second = 'Crude oil production increased by 25 percent.'
    text = first + '\n\n' + second
    result = segment_page(text, hard_boundaries=[len(first) + 2])
    assert any(row['text'] == second and row['reason'] == 'statistical_period_context_missing' for row in result['quarantined'])
    assert_accounted(text, result)


@pytest.mark.parametrize('boundaries', [[True], [0], [9999], [3, 3], [9, 3], ['3'], {}])
def test_invalid_native_boundaries_fail_closed(boundaries):
    with pytest.raises(ValueError, match='Source hard boundaries'):
        segment_page('Crude oil output was stable in 2027.', hard_boundaries=boundaries)


def test_question_only_context_above_safety_maximum_is_accounted_not_counted():
    question = 'Will crude oil production increase in ' + ' '.join(['2027'] * 90) + '?'
    text = ' '.join([question] * 6)
    result = segment_page(text)
    assert not result['accepted']
    assert_accounted(text, result)


@pytest.mark.parametrize('text', [
    'EV purchase prices reach parity with gasoline-powered vehicles in 2029.',
    'EV car prices are within three percent of the respective gasoline vehicle purchase price.',
    'The firm repairs self-propelled diesel equipment.',
    'The company produces diesel batteries and electronic components.',
    'The finance network includes the National Iranian Oil Company.',
])
def test_noncommodity_fuel_comparator_equipment_and_proper_name_are_not_oil_anchors(text):
    assert not oil_scope_supported(text)
    assert not segment_page(text)['accepted']


@pytest.mark.parametrize('text', [
    'EV adoption reduced gasoline consumption by 200,000 barrels per day in 2024.',
    'Diesel equipment consumed 200 barrels of diesel fuel in 2024.',
    'The National Iranian Oil Company exported 200,000 barrels of crude oil in 2024.',
    'OFAC designated the National Iranian Oil Company for Iranian crude oil exports.',
    'Gazprom Neft operates as an oil producer and refiner.',
    'A gasoline-powered fleet consumed 500,000 barrels of gasoline per day in 2024.',
])
def test_masking_noncommodity_phrases_preserves_actual_fuel_volume_trade_evidence(text):
    assert oil_scope_supported(text)
    assert segment_page(text)['accepted']


def test_regional_facility_subset_cannot_inherit_wider_anaphoric_project_total():
    facility = 'New refining projects in the Middle East include a 400,000 barrel per day refinery in Saudi Arabia and a 615,000 barrel per day refinery in Kuwait.'
    total = 'These estimates do not include all refinery expansions. Although delays are possible, these projects could account for nearly 3.0 million barrels per day by 2023.'
    text = facility + '\n\n' + total
    result = segment_page(text)
    assert result['accepted'][0]['text'] == facility
    assert result['quarantined'][0]['text'] == total
    assert result['quarantined'][0]['reason'] == 'aggregate_project_cohort_context_missing'
    assert_accounted(text, result)


def test_explicit_complete_local_project_cohort_and_total_can_bind():
    text = 'Two refinery projects in Asia and the Middle East each add 100,000 barrels per day. These projects together could add 200,000 barrels per day in 2023.'
    result = segment_page(text)
    assert result['accepted'][0]['text'] == text
    assert_accounted(text, result)


def test_facility_facts_before_unqualified_total_in_same_paragraph_are_not_discarded():
    facility = 'A crude oil refinery in Kuwait will add 615,000 barrels per day in 2023.'
    total = 'These projects could collectively add nearly 3.0 million barrels per day in 2023.'
    text = facility + ' ' + total
    result = segment_page(text)
    assert result['accepted'][0]['text'] == facility
    assert result['quarantined'][0]['text'] == total
    assert_accounted(text, result)


def test_native_temporal_question_and_two_contiguous_answers_are_one_exact_context():
    question = "Which of the following is your firm's primary goal in 2027?"
    first = question + ' Executives from exploration and production firms expect to grow production in 2027.'
    second = 'Small firms produced less than 10,000 barrels per day. Their most selected goal was grow production, while larger firms selected reduce debt.'
    text = first + '\n\n' + second
    result = segment_page(text, required_contexts=[dict(start=0, end=len(text), question=question)])
    assert len(result['accepted']) == 1
    assert result['accepted'][0]['text'] == text
    assert 'source_native_question_context' in result['accepted'][0]['exceptions']
    assert not result['quarantined']
    assert_accounted(text, result)


def test_metadata_question_is_not_copied_into_missing_actual_text():
    question = 'What are your crude oil production plans for 2027?'
    text = 'Small firms produced less than 10,000 barrels per day. Their primary goal is grow production.'
    result = segment_page(text, required_contexts=[dict(start=0, end=len(text), question=question)])
    assert not result['accepted']
    assert result['quarantined'][0]['text'] == text
    assert result['quarantined'][0]['reason'] == 'source_question_temporal_context_missing'
    assert_accounted(text, result)


def test_question_context_over_safety_limit_is_held_without_padding_or_truncation():
    question = 'What are your crude oil production plans for 2027?'
    text = question + ' ' + ' '.join(['Crude oil production at our company is expected to rise in 2027.'] * 45)
    result = segment_page(text, required_contexts=[dict(start=0, end=len(text), question=question)])
    assert not result['accepted']
    assert result['quarantined'][0]['reason'] == 'source_question_context_above_safety_maximum'
    assert_accounted(text, result)


def test_required_question_context_cannot_cross_an_independent_speaker():
    question = 'What are your crude oil plans for 2027?'
    text = question + '\n\nAnother respondent expects more crude oil output.'
    with pytest.raises(ValueError, match='independent source boundary'):
        segment_page(text, hard_boundaries=[len(question) + 2],
                     required_contexts=[dict(start=0, end=len(text), question=question)])


@pytest.mark.parametrize('contexts', [{}, [dict(start=True, end=20, question='Oil?')],
    [dict(start=0, end=9999, question='Oil?')], [dict(start=0, end=20, question='Oil')],
    [dict(start=10, end=20, question='Oil?'), dict(start=0, end=10, question='Oil?')]])
def test_invalid_required_contexts_fail_closed(contexts):
    with pytest.raises(ValueError, match='Required source contexts'):
        segment_page('Crude oil production is stable in 2027.', required_contexts=contexts)


def test_question_context_does_not_waive_actual_contact_furniture():
    question = 'What are your oil plans for 2027?'
    text = question + ' Crude oil output will increase in 2027. Contact research@example.org.'
    result = segment_page(text, required_contexts=[dict(start=0, end=len(text), question=question)])
    assert not result['accepted']
    assert result['quarantined'][0]['reason'] == 'report_or_contact_furniture'
    assert_accounted(text, result)


def test_initial_geographic_coordinate_suffix_is_not_a_standalone_short_fact():
    tail = 'TX, and Lafourche Parish, LA, are more closely connected to the offshore oil and gas industry than others.'
    following = 'Crude oil leases are active in Southern California.'
    text = tail + '\n\n' + following
    result = segment_page(text)
    assert result['accepted'][0]['text'] == following
    assert result['quarantined'][0]['text'] == tail
    assert result['quarantined'][0]['reason'] == 'clipped_coordinate_subject_at_initial_edge'
    assert_accounted(text, result)


def test_complete_named_geographic_subject_is_not_rejected_by_coordinate_edge_check():
    text = 'Harris County, TX, and Lafourche Parish, LA, are connected to the offshore oil and gas industry.'
    assert segment_page(text)['accepted'][0]['text'] == text
