"""Native chapter ownership: no mixed public comments/agency answers as prose."""
import pytest

from oil_evidence import unified_corpus as uc
from oil_evidence.store import digest


def appendix(letter='Q', title='RESPONSES TO PUBLIC COMMENTS ON THE DRAFT OIL EIS'):
    return [f'APPENDIX {letter}\n{title}',
            f'{title.title()} {letter}-2\nThe oil lease report received public comments.',
            f'{title.title()} {letter}-3\nCommenter ID Number Comment Response\nA company opposes oil leases. The agency will review the request.',
            f'{title.title()} {letter}-4\nContinued response text mentions crude oil but lacks a commenter and column labels.',
            f'APPENDIX {chr(ord(letter)+1)}\nOIL LEASING ASSESSMENT',
            f'Oil Leasing Assessment {chr(ord(letter)+1)}-2\nCrude oil production at the named field is expected to rise during 2027. The government report describes the expected changes in refinery capacity and petroleum exports.']


def test_whole_native_comment_appendix_includes_continuations_without_column_labels():
    pages = ['Crude oil production at the named field increased during the reporting year 2027. The government report describes the observed changes in refinery capacity and petroleum exports.'] + appendix()
    excluded = uc.native_section_exclusions(pages)
    assert set(excluded) == {2, 3, 4, 5}
    assert all(row['reason'] == 'native_public_comments_and_responses_appendix' for row in excluded.values())
    proof = excluded[5]['native_section']
    assert proof['first_physical_page'] == 2 and proof['last_physical_page'] == 5
    assert proof['appendix_letter'] == 'Q' and proof['native_printed_page'] == 4
    assert proof['next_appendix_physical_page'] == 6 and proof['next_appendix_letter'] == 'R'
    assert proof['cover_raw_page_sha256'] == digest(pages[1])
    assert proof['validation_errors'] == []
    assert uc.page_exclusion(dict(source='unseen_publisher'), pages[4], 5, True, excluded).startswith('native_public_comments')
    assert uc.page_exclusion(dict(source='unseen_publisher'), pages[0], 1, True, excluded) is None
    assert 6 not in excluded and 7 not in excluded
    assert uc.page_exclusion(dict(source='unseen_publisher'), pages[6], 7, True, excluded) is None


def test_native_references_appendix_is_not_admitted_by_oil_in_bibliographic_title():
    pages = appendix('D', 'REFERENCES')
    excluded = uc.native_section_exclusions(pages)
    assert set(excluded) == {1, 2, 3, 4}
    assert all(row['reason'] == 'native_references_appendix' for row in excluded.values())
    assert excluded[4]['native_section']['validation'] == 'verified_continuous_native_running_headers'


@pytest.mark.parametrize('change', ['missing_header', 'wrong_printed_page', 'missing_next_cover', 'wrong_next_letter'])
def test_strong_native_section_cue_with_bad_boundary_fails_closed_without_holding_preceding_body(change):
    pages = appendix()
    if change == 'missing_header':
        pages[2] = 'Comment continuation: oil supply should increase, according to the commenter.'
    elif change == 'wrong_printed_page':
        pages[2] = pages[2].replace('Q-3', 'Q-99')
    elif change == 'missing_next_cover':
        pages[4] = 'The oil assessment continues without a verified appendix cover.'
    else:
        pages[4] = pages[4].replace('APPENDIX R', 'APPENDIX S')
    excluded = uc.native_section_exclusions(['Crude oil imports rose in 2027.'] + pages)
    assert 1 not in excluded
    assert excluded[4]['reason'].endswith('_boundary_validation_failed')
    assert excluded[4]['native_section']['validation_errors']
    # A real subsequent cover remains outside the rejected source section.
    if change != 'missing_next_cover':
        assert 6 not in excluded and 7 not in excluded


def test_final_reference_appendix_can_be_verified_to_physical_eof():
    pages = ['APPENDIX X\nREFERENCES', 'References X-2\nA study of oil supply.', 'References X-3\nAnother oil study.']
    excluded = uc.native_section_exclusions(pages)
    assert set(excluded) == {1, 2, 3}
    assert not excluded[3]['native_section']['validation_errors']
    assert excluded[3]['native_section']['next_appendix_physical_page'] is None


@pytest.mark.parametrize('body', [
    'The oil report references Appendix L responses to public comments in its discussion.',
    'Appendix L responses to public comments were considered when planning oil lease sales.',
    'APPENDIX L\nRESPONSES TO PUBLIC COMMENTS ON THE DRAFT OIL EIS ........ 70',
    'APPENDIX L\nRESPONSES TO PUBLIC COMMENTS ON THE DRAFT OIL EIS\nOil production may increase in 2027.',
    'References A-2 describe how crude oil production increased in 2027.',
    'Responses to Public Comments Q-3 explain a change in oil policy.',
])
def test_prose_mentions_and_toc_are_not_native_covers_or_running_headers(body):
    assert uc.native_section_exclusions([body]) == {}
    assert uc.page_exclusion(dict(source='unseen'), body, 1, True) not in {
        'native_reference_running_header_page', 'native_public_comment_response_running_header_page'}


@pytest.mark.parametrize('heading', ['References A-31', 'REFERENCES 42', 'Bibliography vii', 'Literature Cited D-4'])
def test_strict_plain_reference_running_header_holds_entire_page(heading):
    body = heading + '\nAuthor J. 2023. Oil spill risk analysis. Government report.'
    assert uc.page_exclusion(dict(source='unseen'), body, 1, True) == 'native_reference_running_header_page'


def test_non_pdf_html_is_not_a_native_pdf_appendix():
    assert uc.native_section_exclusions(appendix(), is_pdf=False) == {}


def test_native_trailing_reference_subsection_holds_mixed_start_and_all_continuations():
    pages = ['APPENDIX T\nOIL LEASING METHODS',
             'Oil Leasing Methods T-2\nCrude oil production at the field will increase during 2027.',
             'Oil Leasing Methods T-3\nThe oil report discusses permitting changes.\nT.9 REFERENCES\nAuthor J. 2027. Oil lease activities. Government report.',
             'Oil Leasing Methods T-4\nAuthor K. 2027. Petroleum spill analysis. Government report.',
             'APPENDIX U\nOIL MARKETS',
             'Oil Markets U-2\nThe refinery will increase crude oil processing during 2027.']
    excluded = uc.native_section_exclusions(pages)
    assert set(excluded) == {3, 4}
    assert all(row['reason'] == 'native_reference_subsection' for row in excluded.values())
    proof = excluded[3]['native_section']
    assert proof['native_reference_heading'] == 'T.9 REFERENCES'
    assert proof['whole_page_conservative_hold'] is True
    assert proof['first_physical_page'] == 3 and proof['last_physical_page'] == 4
    assert proof['validation_errors'] == []
    assert 2 not in excluded and 5 not in excluded and 6 not in excluded


def test_native_reference_subsection_stops_before_actual_same_appendix_peer_heading():
    pages = ['APPENDIX T\nOIL LEASING METHODS',
             'Oil Leasing Methods T-2\nT.3 REFERENCES\nAuthor J. 2027. Oil lease activities. Government report.',
             'Oil Leasing Methods T-3\nT.4 CONCLUSIONS\nThe refinery will increase crude oil processing during 2027.']
    excluded = uc.native_section_exclusions(pages)
    assert set(excluded) == {2}
    assert excluded[2]['native_section']['validation_errors'] == []


def test_unknown_nonreference_tail_after_subsection_is_held_but_not_claimed_continuous_reference_pagination():
    pages = ['APPENDIX T\nOIL LEASING METHODS',
             'Oil Leasing Methods T-2\nT.9 REFERENCES\nAuthor J. 2027. Oil lease activities. Government report.',
             'Oil Leasing Methods T-3\nAuthor K. 2027. Petroleum spill analysis. Government report.',
             'The Department of the Interior Mission\nThe department manages natural resources for the nation.']
    excluded = uc.native_section_exclusions(pages)
    assert set(excluded) == {2, 3, 4}
    assert excluded[4]['native_section']['validation_errors']
    assert excluded[4]['native_section']['end_boundary_kind'] == 'physical_eof_or_unverified_tail'
    assert excluded[4]['native_section']['native_printed_page'] is None


def test_numbered_reference_heading_in_toc_or_prose_does_not_start_a_subsection():
    pages = ['APPENDIX T\nOIL LEASING METHODS',
             'Oil Leasing Methods T-2\nT.9 REFERENCES ........ 70\nThe oil report references T.9 REFERENCES when citing scientific studies.']
    assert uc.native_section_exclusions(pages) == {}


def test_reference_subsection_with_missing_native_running_header_is_held_fail_closed():
    pages = ['APPENDIX T\nOIL LEASING METHODS',
             'Oil Leasing Methods T-2\nT.9 REFERENCES\nAuthor J. 2027. Oil lease activities. Government report.',
             'Author K. 2027. Petroleum spill analysis. Government report.',
             'APPENDIX U\nOIL MARKETS']
    excluded = uc.native_section_exclusions(pages)
    assert set(excluded) == {2, 3}
    assert excluded[3]['reason'] == 'native_reference_subsection_boundary_validation_failed'
    assert excluded[3]['native_section']['validation_errors']
