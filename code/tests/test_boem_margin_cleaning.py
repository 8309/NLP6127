from pathlib import Path
import importlib.util
import sys


def module():
    scripts = Path(__file__).parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('boem_margin', scripts / 'collect_boem_supplement.py')
        result = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(result)
        return result
    finally:
        sys.path.remove(str(scripts))


def test_only_observed_repeated_margin_report_furniture_removed():
    m = module()
    body = 'Oil production was 100 barrels per day in 2022.\nThe approved policy does not predict realized future output.'
    pages = ['National OCS Oil and Gas Leasing Program\n' + body + '\nUSDOI | BOEM'] * 4
    output, receipts = m.remove_margin_furniture(pages)
    assert output == [body] * 4
    assert len(receipts) == 4 and all(len(r['removed_lines']) == 2 for r in receipts)
    assert receipts[0]['physical_page'] == 1


def test_body_occurrence_and_factual_statement_never_removed():
    m = module()
    factual = 'National OCS oil and gas production was 100 barrels per day in 2022.'
    header = 'National OCS Oil and Gas Leasing Program'
    pages = ['First line\nSecond line\nThird line\nBody prefix one\nBody prefix two\n' + header + '\n' + factual + '\nBody suffix\nPenultimate line\nLast line'] * 4
    output, receipts = m.remove_margin_furniture(pages)
    assert output == pages
    assert receipts == []


def test_nonreport_units_or_unique_source_heading_preserved():
    m = module()
    pages = ['Million barrels per day\nOil production was 100 barrels per day in 2022.'] * 4
    output, receipts = m.remove_margin_furniture(pages)
    assert output == pages and not receipts


def test_observed_boem_split_header_and_reordered_footer_are_audited():
    m = module()
    pages = ['2023\n2028 National OCS Oil and Gas Leasing Program Draft Programmatic Environmental Impact Statement\n'
             'Chapter 1: Introduction 4 USDOI | BOEM\nOil production was 100 barrels per day in 2022.'] * 4
    output, receipts = m.remove_margin_furniture(pages)
    assert output == ['Oil production was 100 barrels per day in 2022.'] * 4
    assert len(receipts[0]['removed_lines']) == 3
    assert receipts[0]['removed_lines'][0]['reason'] == 'split_year_of_verified_repeated_report_header'


def test_verified_footer_may_vary_in_page_number_without_body_deletion():
    m = module()
    pages = [f'Chapter 1: Introduction {i} USDOI | BOEM\nOil output was 100 barrels per day in 2022.' for i in range(4)]
    output, receipts = m.remove_margin_furniture(pages)
    assert output == ['Oil output was 100 barrels per day in 2022.'] * 4
    assert all(r['removed_lines'][0]['reason'] == 'visually_verified_page_numbered_BOEM_margin_footer' for r in receipts)
