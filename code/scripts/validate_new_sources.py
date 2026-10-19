"""Conservative version-date audit, touching only this run's new metadata."""
import argparse
import json
import re
from datetime import datetime
from bs4 import BeautifulSoup
from pypdf import PdfReader
from expand_corpus import select_run,snapshot,verify_baseline_sources,pipeline_lock
from oil_evidence.store import DATA,read_jsonl,write_jsonl,write_json,now,digest


def pdf_date(value):
    m=re.match(r'(?:D:)?(20\d{2})(\d{2})(\d{2})',str(value or ''))
    if not m:
        return None
    try:
        return datetime(int(m[1]),int(m[2]),int(m[3])).date().isoformat()
    except ValueError:
        return None


def validate(run):
    root=select_run(run)
    snapshot()
    verify_baseline_sources()
    old={r['document_id'] for r in read_jsonl(root/'baseline/documents.jsonl')}
    docs=read_jsonl(DATA/'documents.jsonl')
    sources={r['url']:r for r in read_jsonl(DATA/'sources.jsonl')}
    records=[]
    for d in docs:
        if d['document_id'] in old:
            continue
        raw=(DATA/d['raw_path']).read_bytes()
        assert digest(raw)==d['raw_sha256']
        record=dict(document_id=d['document_id'],source=d['source'],url=d['url'],available_by=d['available_by'])
        if raw.startswith(b'%PDF'):
            reader=PdfReader(DATA/d['raw_path'])
            metadata=reader.metadata or {}
            cover='\n'.join(p.extract_text() or '' for p in reader.pages[:2])
            record.update(pages=len(reader.pages),cover_excerpt=cover[:1800],
                file_creation=pdf_date(metadata.get('/CreationDate')),file_modified=pdf_date(metadata.get('/ModDate')))
            if d.get('requires_pdf'):
                latest=max((v for v in [record['file_creation'],record['file_modified']] if v),default=None)
                # A file produced later than its latest visible release/update cannot
                # substantiate historical availability. Retain raw, quarantine prose.
                if latest and latest>d['available_by']:
                    d['ingest_exclusion_reason']='Current PDF file date is later than visible publication/revision; historical attachment version unverified'
                    record['excluded']=d['ingest_exclusion_reason']
            if d['source']=='DOE-SPR':
                evidence=d.get('date_evidence',{})
                if evidence.get('cover_quote') and not re.sub(r'\s+',' ',evidence['cover_quote']).lower() in re.sub(r'\s+',' ',cover).lower():
                    # OCR/text-layer formatting may differ: report this for inspection,
                    # rather than silently substituting the report's calendar year.
                    record['cover_date_check']='needs_text_layer_inspection'
        elif d['source']=='BSEE-HURRICANE':
            soup=BeautifulSoup(raw,'html.parser')
            body=soup.select_one('main article .field--name-body')
            text=body.get_text(' ',strip=True) if body else ''
            heading=soup.select_one('main h1')
            version_text=text+' '+(heading.get_text(' ',strip=True) if heading else '')+' '+d.get('title','')
            if body is None:
                d['ingest_exclusion_reason']='No trusted BSEE body selector; fail closed on version and body extraction'
                record['excluded']=d['ingest_exclusion_reason']
            elif re.search(r'Gulf of America',version_text,re.I) and d['available_by']<'2025-01-20':
                # Renaming proves this is not an unchanged historic snapshot, but
                # does not prove when this current body was uploaded. Do not invent
                # a January 2025 publication day/month; retain raw and exclude prose.
                d['historical_attachment_version']='Current body uses post-January-2025 terminology; unchanged historical body not established'
                d['ingest_exclusion_reason']='Current BSEE body renamed after original publication; historical version unverified'
                record['excluded']=d['ingest_exclusion_reason']
        sources[d['url']].update({k:v for k,v in d.items() if k in {
            'available_by','published_date','date_precision','date_source','original_date_metadata',
            'historical_attachment_version','ingest_exclusion_reason'}})
        records.append(record)
    write_jsonl(DATA/'documents.jsonl',docs)
    write_jsonl(DATA/'sources.jsonl',sources.values())
    report=dict(at=now(),new_documents=len(records),excluded=sum('excluded' in r for r in records),
        adjusted=sum('date_adjustment' in r for r in records),human_reviewed=False,external_model_calls=0,records=records)
    report.update(documents_sha256=digest((DATA/'documents.jsonl').read_bytes()),
        sources_sha256=digest((DATA/'sources.jsonl').read_bytes()),validator_sha256=digest(__import__('pathlib').Path(__file__).read_bytes()))
    write_json(root/'new_source_version_audit.json',report)
    return {k:v for k,v in report.items() if k!='records'}


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    args=p.parse_args()
    select_run(args.run)
    with pipeline_lock():
        print(json.dumps(validate(args.run),indent=2))
