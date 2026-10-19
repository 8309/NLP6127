"""Actual linked EIA oil-supply methodology and outlook series, no guessed URLs."""
from urllib.parse import urljoin, urlparse
import re
from bs4 import BeautifulSoup
from oil_evidence.corpus import date_from_text

FINANCE = 'https://www.eia.gov/finance/review/'
METHODS = 'https://www.eia.gov/outlooks/aeo/nems/documentation/'
IEO = 'https://www.eia.gov/outlooks/ieo/ieoarchive.php'


def candidate(url, title, kind, date, origin, precision='day', **extra):
    if not date or not '2018-01-01' <= date <= '2026-10-05':
        return None
    if urlparse(url).hostname != 'www.eia.gov' or not urlparse(url).path.lower().endswith('.pdf'):
        return None
    return dict(url=url,title=title,source=kind,published_date=date if precision=='day' else None,
        available_by=date,date_precision=precision,date_source=origin,event_date=None,event_status='unlabelled',
        redistribution='eia_attribution',requires_pdf=True,**extra)


def methods_sources(raw, origin=METHODS):
    soup=BeautifulSoup(raw,'html.parser')
    rows=[]
    allowed=('Oil and Gas Supply','Liquids Fuels Market','International Energy')
    for a in soup.select('a[href]'):
        year=a.get_text(' ',strip=True)
        if year not in {'2018','2020','2022','2023','2025'}:
            continue
        group=next((u.get_text(' ',strip=True) for u in a.find_parents('ul')
                    if u.get_text(' ',strip=True).startswith(allowed)),None)
        if not group:
            continue
        module=next(t for t in allowed if group.startswith(t))
        row=candidate(urljoin(origin,a['href']),f'EIA {module} Model Documentation {year}',
            'EIA-METHODS',year+'-12-31',origin,'year_upper_bound',content_scope='oil_supply_model_mechanisms',
            reporting_year=int(year),date_evidence='Explicit edition year in official archive; conservative year end')
        if row:
            rows.append(row)
    return rows


def finance_sources(raw, origin=FINANCE):
    soup=BeautifulSoup(raw,'html.parser')
    rows=[]
    for a in soup.select('a[href]'):
        context=a.find_parent(['tr','li'])
        if not context:
            continue
        # Invalid archive HTML nests later entries in one li; use only text before
        # this PDF anchor, never a later edition's date.
        prefix=[]
        for node in context.descendants:
            if node is a:
                break
            if isinstance(node,str):
                prefix.append(str(node))
        text=' '.join(prefix)
        exact=date_from_text(text)
        title=text.strip()
        if not exact or len(title)>180:
            continue
        reporting=re.search(r'\b(20\d{2})\b',title)
        # An observed Annual 2020 archive entry claims May 2020: fail closed,
        # rather than publish a full reporting-year result before the year ends.
        if re.search(r'Annual',title,re.I) and reporting and exact[:4]<=reporting[1]:
            continue
        row=candidate(urljoin(origin,a['href']),'EIA Financial Review: '+title,
            'EIA-FINANCE',exact,origin,content_scope='oil_company_production_investment')
        if row:
            rows.append(row)
    return rows


def ieo_sources(raw, origin=IEO):
    soup=BeautifulSoup(raw,'html.parser')
    rows=[]
    for a in soup.select('a[href]'):
        href=urljoin(origin,a['href'])
        m=re.search(r'/outlooks/archive/ieo(18|19|20)/pdf/',href)
        if not m:
            continue
        year='20'+m[1]
        row=candidate(href,f'EIA International Energy Outlook {year}',
            'EIA-IEO',year+'-12-31',origin,'year_upper_bound',content_scope='oil_narratives_only')
        if row:
            rows.append(row)
    return list({r['url']:r for r in rows}.values())


def discover(fetch):
    rows={}
    for url,parser in [(METHODS,methods_sources),(FINANCE,finance_sources),(IEO,ieo_sources)]:
        try:
            raw,final,_=fetch.get(url)
            found=parser(raw,final)
            rows.update((r['url'],r) for r in found)
            print(f'Discovered {len(found)} EIA series candidates from {final}',flush=True)
        except Exception as e:
            print(f'Skipped {url}: {e}',flush=True)
    return list(rows.values())
