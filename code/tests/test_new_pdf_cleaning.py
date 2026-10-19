from oil_evidence.corpus import clean_page,has_table_heading,is_reference_page


def test_wrapped_year_is_not_a_chart_tick():
    assert 'demand in 2019.' in clean_page('The report compares petroleum demand in\n2019.\nIt discusses crude oil production.')
    assert clean_page('Chart 3.3\n2019\n2020\n2050')=='Chart 3.3'


def test_table_and_reference_continuations():
    assert has_table_heading('Table A-3. Refinery capacities')
    assert has_table_heading('Table 7-1 Costs')
    assert is_reference_page('List of cited works.\nReferences 9-2 September 2023')
