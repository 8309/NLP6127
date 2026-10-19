import importlib.util
import json
from pathlib import Path
import sys


def load_module():
    scripts = Path(__file__).parents[1] / 'scripts'
    sys.path.insert(0, str(scripts))
    try:
        spec = importlib.util.spec_from_file_location('treasury_buffer', scripts / 'discover_treasury_buffer.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(scripts))


def test_broader_country_sanctions_requires_body_not_fictional_oil_title():
    module = load_module()
    items = [{'date': '2024-02-23', 'title': 'Treasury Sanctions Hundreds of Targets in Russia', 'url': '/news/press-releases/jy2117'},
             {'date': '2020-01-01', 'title': 'Treasury Sanctions Domestic Bank', 'url': '/news/press-releases/x1'}]
    rows = module.candidates(json.dumps({'items': items}).encode(), module.TREASURY, set())
    assert len(rows) == 1
    assert rows[0]['title_match'] == 'country_sanctions_body_check_required'
    assert rows[0]['title'] == items[0]['title']
    assert rows[0]['event_date'] is None


def test_window_known_links_and_deceptive_hosts_are_excluded():
    module = load_module()
    items = [{'date': day, 'title': 'Treasury Sanctions Iran Oil', 'url': url} for day, url in [
        ('2017-12-31', '/news/press-releases/x1'), ('2026-10-06', '/news/press-releases/x2'),
        ('2024-01-01', 'https://home.treasury.gov.evil.test/news/press-releases/x3'),
        ('2024-01-01', 'https://home.treasury.gov:444/news/press-releases/x4'),
        ('2024-01-01', '/news/press-releases/x5'), ('not-a-date', '/news/press-releases/x6')]]
    known = {'https://home.treasury.gov/news/press-releases/x5'}
    assert module.candidates(json.dumps({'items': items}).encode(), module.TREASURY, known) == []
