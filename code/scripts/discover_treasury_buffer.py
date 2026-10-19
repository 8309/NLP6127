"""Immutable broader Treasury discovery; title candidates are not oil evidence.

Some country-sanctions releases contain substantive oil actions without naming
shipping/trade in the title. Follow the actual linked official search shards,
then require dated, mineral-oil body evidence at collection and segmentation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlparse

from discover_unified_sources import (TREASURY, START, END, DIRECT_OIL, COUNTRY,
    SANCTIONS, canonical, official, parse_manifest_link, parse_shard_links,
    parse_treasury_body, frozen_live_hashes)
from oil_evidence.corpus import Fetcher
from oil_evidence.segmentation import oil_scope_supported
from oil_evidence.store import DATA, digest, now, read_jsonl, words, write_json, write_jsonl


def candidates(raw, origin, known):
    payload = json.loads(raw)
    items = payload.get('items')
    if not isinstance(items, list):
        raise ValueError('Official search shard has no item list')
    rows = []
    for item in items:
        exact, title = item.get('date'), item.get('title')
        try:
            if not isinstance(exact, str) or date.fromisoformat(exact).isoformat() != exact:
                continue
        except ValueError:
            continue
        if not START <= exact <= END or not isinstance(title, str) or not title.strip():
            continue
        direct = bool(DIRECT_OIL.search(title))
        broad = bool(COUNTRY.search(title) and SANCTIONS.search(title))
        if not (direct or broad):
            continue
        href = urljoin(origin, item.get('url', ''))
        p = urlparse(href)
        if not official(href, 'home.treasury.gov') or p.port not in {None, 443}:
            continue
        if not re.fullmatch(r'/news/press-releases/[a-z0-9-]+/?', p.path) or canonical(href) in known:
            continue
        rows.append(dict(url=href, source='TREASURY-OIL', title=title,
            published_date=exact, available_by=exact, date_precision='day',
            date_source=origin, event_date=None, event_status='unlabelled',
            redistribution='review_required', requires_pdf=False,
            content_scope='oil_actions_only_after_body_and_paragraph_check',
            discovery_method='actually_linked_official_shard_broader_country_sanctions_screen',
            date_evidence='Official indexed publication day; body publication must agree',
            title_match='direct_oil' if direct else 'country_sanctions_body_check_required'))
    return rows


def discover(output, pilot_limit=6):
    output = Path(output).resolve()
    if any((output / name).exists() for name in ['source_candidates.jsonl', 'discovery_report.json']):
        raise ValueError('Discovery outputs are immutable; choose a new directory')
    known = {canonical(d['url']) for d in read_jsonl(DATA / 'documents.jsonl')}
    for run in ['semantic-page-v1', 'semantic-page-v2-buffer']:
        known.update(canonical(d['url']) for d in read_jsonl(DATA / 'unified_runs' / run / 'supplement/documents.jsonl'))
    before = frozen_live_hashes()
    archives, pilots, errors, found = [], [], [], {}
    fetch = Fetcher()
    try:
        raw, origin, _ = fetch.get(TREASURY)
        manifest_url = parse_manifest_link(raw, origin)
        raw, origin, _ = fetch.get(manifest_url)
        for shard in parse_shard_links(raw, origin):
            raw, final, _ = fetch.get(shard)
            rows = candidates(raw, final, known)
            found.update((row['url'], row) for row in rows)
            archives.append(dict(url=final, raw_sha256=digest(raw), bytes=len(raw), new_title_candidates=len(rows)))
            print(f'Treasury broader title candidates: {len(rows)} in {final}', flush=True)
        rows = sorted(found.values(), key=lambda r: (r['available_by'], r['url']))
        indices = sorted({round(i * (len(rows) - 1) / max(1, pilot_limit - 1)) for i in range(pilot_limit)}) if rows else []
        for i in indices:
            row = rows[i]
            record = dict(url=row['url'], title=row['title'], indexed_date=row['available_by'])
            try:
                raw, final, _ = fetch.get(row['url'])
                if not official(final, 'home.treasury.gov'):
                    raise ValueError('Pilot redirected away from official host')
                exact, body, blocks = parse_treasury_body(raw)
                record.update(article_date=exact, date_matches=exact == row['available_by'],
                    strong_oil_body=oil_scope_supported(body), body_words=len(words(body)),
                    factual_blocks=len(blocks), raw_sha256=digest(raw), body_sha256=digest(body), body_preview=body[:500])
                record['status'] = 'dated_oil_body' if record['date_matches'] and record['strong_oil_body'] else 'scope_or_date_excluded'
            except Exception as exc:
                record.update(status='failed_no_bypass', error=str(exc))
            pilots.append(record)
    except Exception as exc:
        errors.append(dict(error=str(exc), stage='linked_archive_discovery'))
        rows = sorted(found.values(), key=lambda r: (r['available_by'], r['url']))
    finally:
        response_bytes = fetch.bytes
        fetch.close()
    if before != frozen_live_hashes():
        raise ValueError('Live input changed during discovery')
    write_jsonl(output / 'source_candidates.jsonl', rows)
    report = dict(at=now(), discovery_only=True, window=[START, END], title_candidates=len(rows),
        by_year=dict(sorted(Counter(r['available_by'][:4] for r in rows).items())),
        archives=archives, fixed_date_spread_pilots=pilots, errors=errors,
        candidate_file_sha256=digest((output / 'source_candidates.jsonl').read_bytes()),
        script_sha256=digest(Path(__file__).read_bytes()), raw_documents_written=0,
        external_model_calls=0, live_inputs_unchanged=True, downloaded_response_bytes=response_bytes,
        limitations=['A title match is only a collection candidate, not oil evidence or an accepted passage.',
            'Fixed date-spread pilots are not a representative yield estimate.',
            'Country sanctions without strong oil body evidence are retained but excluded by the collector.',
            'Attribution, legal effect, announcement versus implementation and redistribution remain separately reviewed.'])
    write_json(output / 'discovery_report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pilot-limit', type=int, choices=range(0, 9), default=6)
    args = parser.parse_args()
    result = discover(args.output_dir, args.pilot_limit)
    print(json.dumps({k: result[k] for k in ['title_candidates', 'by_year', 'errors', 'candidate_file_sha256']}, indent=2))
