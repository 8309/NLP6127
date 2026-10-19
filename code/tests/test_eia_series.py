import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from discover_eia_series import finance_sources,methods_sources,ieo_sources


def test_finance_actual_date_not_reporting_year():
    raw='<table><tr><th>Annual 2023 (6/3/2024)</th><td><a href="/finance/review/archive/pdf/2023.pdf">PDF</a></td></tr></table>'
    assert finance_sources(raw)[0]['available_by']=='2024-06-03'


def test_finance_ambiguous_bad_date_fails_closed():
    raw='<li>Annual 2020 (5/27/2020)<a href="/finance/review/archive/pdf/2020.pdf">PDF</a></li>'
    assert not finance_sources(raw)


def test_methods_only_oil_related_modules():
    raw='<ul>Oil and Gas Supply<li><a href="archive/pdf/m063(2020).pdf">2020</a></li></ul><ul>Macroeconomic Activity<li><a href="archive/pdf/m065(2020).pdf">2020</a></li></ul>'
    assert len(methods_sources(raw))==1
    assert methods_sources(raw)[0]['available_by']=='2020-12-31'


def test_ieo_foreign_host_and_future_ignored():
    raw='<a href="https://bad.org/outlooks/archive/ieo20/pdf/x.pdf">PDF</a><a href="/outlooks/archive/ieo19/pdf/ieo2019.pdf">PDF</a>'
    assert len(ieo_sources(raw))==1
