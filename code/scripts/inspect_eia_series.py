"""Read-only inspection of observed EIA archives; preserves discovery evidence."""
import argparse
from bs4 import BeautifulSoup
from expand_corpus import RecordingFetcher,select_run

parser=argparse.ArgumentParser()
parser.add_argument('url',nargs='+')
args=parser.parse_args()
select_run('official-2018-2026-v2')
f=RecordingFetcher()
try:
    for url in args.url:
        raw,final,_=f.get(url)
        soup=BeautifulSoup(raw,'html.parser')
        main=soup.select_one('.pagecontent') or soup.select_one('main') or soup
        print('URL:',final)
        for a in main.select('a[href]'):
            href=a['href']
            if any(x in href for x in ['.pdf','/archive','/nems','assumptions']) or a.get_text(strip=True).isdigit():
                parent=a.find_parent(['li','tr']) or a.parent
                print(repr(a.get_text(' ',strip=True)),repr(href),repr(parent.get_text(' ',strip=True)[:320]))
finally:
    f.close()
