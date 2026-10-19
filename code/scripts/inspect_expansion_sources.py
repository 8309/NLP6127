"""Read-only official-page inspection through the rate-limited fetcher."""
import re
from bs4 import BeautifulSoup
from oil_evidence.corpus import Fetcher

fetch = Fetcher()
try:
    raw, _, _ = fetch.get('https://www.eia.gov/international/main-2Q73HVIX.js')
    text = raw.decode()
    print('JS bytes', len(raw))
    for token in ['countries_long', 'analysis_includes', '.json', 'analysis/', 'country-analysis', 'loadChildren', 'path:"analysis"']:
        matches = list(re.finditer(re.escape(token), text))
        print('TOKEN', token, 'COUNT', len(matches))
        for m in matches[:12]:
            print(text[max(0, m.start()-150):m.end()+220])
    raw, _, _ = fetch.get('https://www.eia.gov/international/content/analysis/countries_long/Iran/')
    soup = BeautifulSoup(raw, 'html.parser')
    print('BASE', soup.find('base'))
finally:
    fetch.close()
