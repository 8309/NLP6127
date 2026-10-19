"""Audited local-only repairs of already downloaded, excluded BOEM PDFs.

Never rewrite the original supplement registries or an existing extraction
cache. Positive admissions are append-only stage derivations, not semantic or
human approval. No network or model service is used by this module.
"""
from __future__ import annotations

import argparse
import copy
from datetime import date
import io
import json
from pathlib import Path
import re

import collect_boem_supplement as margin_module
import discover_boem_buffer as parser_module
from collect_unified_supplement import validate_manifest_pair, writer_lock
from oil_evidence.corpus import extraction_suffix
from oil_evidence.store import DATA, append_jsonl, digest, now, read_jsonl, words, write_json


VERSION = 'boem-local-reprocessing-admission-v1'
START, END = '2018-01-01', '2026-10-05'
DOCL_ONLY = {'document_id', 'final_url', 'content_type', 'raw_path', 'raw_sha256', 'bytes', 'fetched_at'}
MUTABLE = {
    'available_by', 'indexed_available_by', 'body_date_verified', 'cover_month',
    'pdf_creation_date', 'pdf_modified_date', 'physical_pages', 'extraction_audit_path',
    'extraction_audit_sha256', 'body_parser', 'body_parser_sha256', 'body_word_count',
    'publication_version_caution', 'body_parser_error', 'ingest_exclusion_reason', 'prior_error',
}
PRIOR_KEYS = ('ingest_exclusion_reason', 'body_parser_error', 'body_date_verified', 'body_parser_sha256')
SHA = re.compile(r'[0-9a-f]{64}')


def canonical_sha(value):
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False))


def _bound_path(data, value, *, under=None):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError('Reprocessing artifacts require nonempty data-relative paths')
    path = (data / value).resolve()
    allowed = data.resolve() if under is None else under.resolve()
    if not path.is_relative_to(allowed) or not path.is_file():
        raise ValueError('Reprocessing artifact escapes its allowed root or is missing')
    return path


def _file_binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path.read_bytes()))


def _implementations():
    return dict(parser=_file_binding(parser_module.__file__),
                margin_cleaner=_file_binding(margin_module.__file__),
                script=_file_binding(__file__))


def _check_implementations(bindings):
    if not isinstance(bindings, dict) or bindings != _implementations():
        raise ValueError('Reprocessing parser/script/margin implementation path or SHA drifted')


def _day(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value or not START <= value <= END:
        raise ValueError('Reprocessing date outside fixed window or invalid canonical day')
    return value


def _prior(record):
    return {key: copy.deepcopy(record[key]) for key in PRIOR_KEYS if key in record}


def _original_pair(doc, source):
    if not isinstance(doc, dict) or not isinstance(source, dict):
        raise ValueError('Reprocessing source/document must be objects')
    if doc.get('source') != 'BOEM-EIS' or not parser_module.official(doc.get('url', '')):
        raise ValueError('Reprocessing is restricted to official BOEM-EIS sources')
    if not parser_module.official(doc.get('final_url', '')):
        raise ValueError('BOEM original final attachment must remain official HTTPS')
    if canonical_sha({key: value for key, value in doc.items() if key not in DOCL_ONLY}) != canonical_sha(source):
        raise ValueError('Original supplemental source/document metadata differs')
    if (doc.get('ingest_exclusion_reason') != 'unverified_pdf_publication_or_extraction'
            or doc.get('body_date_verified') is not False
            or not isinstance(doc.get('body_parser_error'), str) or not doc['body_parser_error'].strip()):
        raise ValueError('Only prior BOEM parser/dependency failures can be positively reprocessed')
    if doc.get('document_id') != digest(doc['url'])[:20]:
        raise ValueError('Original BOEM document identity does not match URL')
    if not SHA.fullmatch(doc.get('raw_sha256', '')):
        raise ValueError('Original BOEM raw SHA is invalid')
    if type(doc.get('bytes')) is not int or not 0 < doc['bytes'] <= 25 * 1024**2:
        raise ValueError('Original BOEM raw byte count exceeds 25 MB or is invalid')
    _day(doc.get('available_by'))


def _raw(data, doc):
    path = _bound_path(data, doc['raw_path'], under=data / 'raw')
    raw = path.read_bytes()
    if len(raw) != doc['bytes'] or digest(raw) != doc['raw_sha256'] or not raw.startswith(b'%PDF'):
        raise ValueError('Original BOEM raw bytes/path/SHA differ')
    return raw


def _pdf_dates(raw):
    metadata = parser_module.PdfReader(io.BytesIO(raw)).metadata or {}
    return (parser_module.pdf_timestamp(metadata.get('/CreationDate')),
            parser_module.pdf_timestamp(metadata.get('/ModDate')))


def _check_derived(original, derived):
    if not isinstance(derived, dict):
        raise ValueError('Derived reprocessing metadata must be an object')
    preserved_original = {key: value for key, value in original.items() if key not in MUTABLE}
    preserved_derived = {key: value for key, value in derived.items() if key not in MUTABLE}
    if canonical_sha(preserved_original) != canonical_sha(preserved_derived):
        raise ValueError('Derived reprocessing changed immutable metadata or added unauthorized fields')
    if 'ingest_exclusion_reason' in derived or 'body_parser_error' in derived:
        raise ValueError('Positive reprocessing must remove only the original failure state')
    if derived.get('prior_error') != _prior(original) or derived.get('body_date_verified') is not True:
        raise ValueError('Derived reprocessing requires exact prior failure and verified date')
    if _day(derived.get('available_by')) < _day(original.get('available_by')):
        raise ValueError('Reprocessing cannot backdate availability')


def apply_reprocessing_admissions(documents, sources, *, data, runroot):
    """Return validated stage-only copies and input hashes; mutate no argument.

    Call before the negative admission overlay. The negative ledger is also
    checked here so a positive repair can never overrule a quality exclusion.
    """
    data, runroot = Path(data).resolve(), Path(runroot).resolve()
    path = runroot / 'supplement/reprocessing_admissions.jsonl'
    bound = {path.relative_to(data).as_posix(): digest(path.read_bytes())} if path.exists() else {}
    return _validate_admission_rows(documents, sources, data=data, runroot=runroot,
                                    admissions=read_jsonl(path), bound=bound)


def _validate_admission_rows(documents, sources, *, data, runroot, admissions, bound=None):
    """Same gate for prospective rows before append and persisted consumer rows."""
    data, runroot = Path(data).resolve(), Path(runroot).resolve()
    if not runroot.is_relative_to(data / 'unified_runs'):
        raise ValueError('Reprocessing run root must be inside data/unified_runs')
    docs, source_rows = copy.deepcopy(documents), copy.deepcopy(sources)
    doc_by_id = {row['document_id']: row for row in docs}
    source_by_url = {row['url']: row for row in source_rows}
    if len(doc_by_id) != len(docs) or len(source_by_url) != len(source_rows):
        raise ValueError('Reprocessing input contains duplicate source/document identities')
    if {row['url'] for row in docs} != set(source_by_url) or len(docs) != len(source_rows):
        raise ValueError('Reprocessing requires paired document/source manifests')
    if not admissions:
        return docs, source_rows, dict(bound or {})
    bound = dict(bound or {})
    negative = runroot / 'supplement/admission_exclusions.jsonl'
    negative_ids = set()
    if negative.exists():
        for row in read_jsonl(negative):
            if not isinstance(row, dict) or not isinstance(row.get('document_id'), str):
                raise ValueError('Negative admission ledger contains invalid identity')
            negative_ids.add(row['document_id'])
        bound[negative.relative_to(data).as_posix()] = digest(negative.read_bytes())
    seen = set()
    for admission in admissions:
        if not isinstance(admission, dict):
            raise ValueError('Reprocessing admission must be an object')
        identifier = admission.get('document_id')
        if identifier not in doc_by_id or identifier in seen:
            raise ValueError('Unknown or duplicate positive reprocessing document ID')
        seen.add(identifier)
        original = doc_by_id[identifier]
        source = source_by_url[original['url']]
        if (identifier in negative_ids
                or str(original.get('ingest_exclusion_reason', '')).startswith('audited_post_collection:')
                or str(source.get('ingest_exclusion_reason', '')).startswith('audited_post_collection:')):
            raise ValueError('Positive reprocessing cannot override audited post-collection negative admission')
        _original_pair(original, source)
        if (admission.get('decision_version') != VERSION
                or admission.get('original_document_sha256') != canonical_sha(original)
                or admission.get('original_source_sha256') != canonical_sha(source)
                or admission.get('raw_sha256') != original['raw_sha256']):
            raise ValueError('Reprocessing admission differs from original canonical objects/raw')
        if (admission.get('model_only') is not True or admission.get('human_gold') is not False
                or admission.get('semantic_approval') is not False):
            raise ValueError('Reprocessing is machine-only version checking, never semantic/human gold approval')
        _check_implementations(admission.get('implementations'))
        derived_doc, derived_source = admission.get('derived_document'), admission.get('derived_source')
        _check_derived(original, derived_doc)
        _check_derived(source, derived_source)
        if canonical_sha({k: v for k, v in derived_doc.items() if k not in DOCL_ONLY}) != canonical_sha(derived_source):
            raise ValueError('Derived reprocessing source/document metadata differs')
        raw = _raw(data, original)
        bound[original['raw_path']] = original['raw_sha256']
        blobs = {}
        for name in ('cache', 'audit', 'source_pages', 'proof'):
            item = admission.get(name)
            if not isinstance(item, dict) or not SHA.fullmatch(item.get('sha256', '')):
                raise ValueError('Reprocessing requires exact artifact path/SHA bindings')
            allowed = data / 'extracted' if name == 'cache' else runroot / 'reports/boem-reprocessing'
            artifact = _bound_path(data, item.get('path'), under=allowed)
            blob = artifact.read_bytes()
            if digest(blob) != item['sha256']:
                raise ValueError('Reprocessing artifact bytes differ from bound SHA')
            bound[item['path']] = item['sha256']
            blobs[name] = json.loads(blob)
        expected_cache = (data / 'extracted' / (original['raw_sha256'] + extraction_suffix(original['source']))).resolve()
        if (data / admission['cache']['path']).resolve() != expected_cache:
            raise ValueError('Positive repair must bind the exact source-bound extraction cache')
        pages, original_pages = blobs['cache'], blobs['source_pages']
        if (not isinstance(pages, list) or not pages or not isinstance(original_pages, list)
                or len(pages) != len(original_pages) or not all(isinstance(p, str) for p in pages + original_pages)):
            raise ValueError('Reprocessing must retain every physical page in original order')
        expected_pages, removed = margin_module.remove_margin_furniture(original_pages)
        audit = blobs['audit']
        expected_audit = dict(raw_sha256=original['raw_sha256'], implementations=admission['implementations'],
            source_pdf_pages_sha256=canonical_sha(original_pages), cache_sha256=admission['cache']['sha256'],
            physical_pages=len(pages), pages_with_removed_margin_lines=removed,
            removal_policy='recognized_exact_repeated_margin_only; no body rewrites or synthetic facts')
        if pages != expected_pages or audit != expected_audit:
            raise ValueError('Reprocessing cache/margin audit cannot replay the source page transformations')
        proof = blobs['proof']
        if (not isinstance(proof, dict) or proof.get('decision_version') != VERSION
                or proof.get('document_id') != identifier or proof.get('raw_sha256') != original['raw_sha256']
                or proof.get('original_document_sha256') != canonical_sha(original)
                or proof.get('original_source_sha256') != canonical_sha(source)
                or proof.get('implementations') != admission['implementations']
                or proof.get('cache') != admission['cache'] or proof.get('audit') != admission['audit']
                or proof.get('source_pages') != admission['source_pages']
                or proof.get('model_only') is not True or proof.get('human_gold') is not False
                or proof.get('semantic_approval') is not False or proof.get('accepted_passages') is not None
                or type(proof.get('external_model_calls')) is not int or proof['external_model_calls'] != 0
                or type(proof.get('human_labels_created')) is not int or proof['human_labels_created'] != 0):
            raise ValueError('Reprocessing machine-only proof is not bound to this exact repair')
        parsed = proof.get('parsed_metadata')
        if not isinstance(parsed, dict) or parsed.get('date_confirmed') is not True:
            raise ValueError('Reprocessing proof lacks verified parser date metadata')
        created, modified = _pdf_dates(raw)
        indexed = source.get('indexed_available_by') or source['available_by']
        valid_timestamps = [value for value in (created, modified) if value]
        if not valid_timestamps:
            raise ValueError('Current PDF metadata dates missing in reprocessing')
        expected_available = _day(max([_day(indexed)] + valid_timestamps))
        month = parser_module.cover_month(original_pages)
        if month > indexed[:7] or not START[:7] <= month <= END[:7]:
            raise ValueError('Reprocessing native cover conflicts with original indexed date/window')
        expected_metadata = dict(published_date=original.get('published_date'), available_by=expected_available,
            indexed_available_by=indexed, cover_month=month, pdf_creation_date=created, pdf_modified_date=modified,
            physical_pages=len(pages))
        if any(parsed.get(k) != v for k, v in expected_metadata.items()):
            raise ValueError('Parser proof disagrees with native cover/current PDF metadata/date')
        updates = dict(available_by=expected_available, indexed_available_by=indexed,
            body_date_verified=True, cover_month=month, pdf_creation_date=created, pdf_modified_date=modified,
            physical_pages=len(pages), extraction_audit_path=admission['audit']['path'],
            extraction_audit_sha256=admission['audit']['sha256'], body_parser=parsed.get('parser'),
            body_parser_sha256=admission['implementations']['parser']['sha256'],
            body_word_count=sum(len(words(page)) for page in pages),
            publication_version_caution=parsed.get('version_caution'))
        if not isinstance(updates['body_parser'], str) or not updates['body_parser'] or not updates['publication_version_caution']:
            raise ValueError('Positive repair requires parser identity and explicit version caution')
        for old, derived in ((original, derived_doc), (source, derived_source)):
            expected = copy.deepcopy(old)
            expected.pop('ingest_exclusion_reason', None)
            expected.pop('body_parser_error', None)
            expected.update(updates, prior_error=_prior(old))
            if canonical_sha(expected) != canonical_sha(derived):
                raise ValueError('Derived repair contains unapproved fields or differs from verified metadata')
        doc_by_id[identifier], source_by_url[original['url']] = copy.deepcopy(derived_doc), copy.deepcopy(derived_source)
    return ([doc_by_id[doc['document_id']] for doc in docs],
            [source_by_url[source['url']] for source in source_rows], bound)


def _new_json(path, value):
    if path.exists():
        raise ValueError('Existing reprocessing artifact/cache cannot be overwritten')
    write_json(path, value)
    return dict(path=path.relative_to(DATA).as_posix(), sha256=digest(path.read_bytes()))


def _protections(data, runroot):
    paths = [runroot / 'supplement/documents.jsonl', runroot / 'supplement/sources.jsonl']
    for directory in ('raw', 'extracted'):
        paths.extend(path for path in (data / directory).rglob('*') if path.is_file())
    paths.extend(data / name for name in (
        'documents.jsonl', 'sources.jsonl', 'passages.jsonl', 'index/manifest.json', 'index/vectors.npy',
        'annotations/codex.jsonl', 'annotations/passage_labels.jsonl', 'annotations/passage_review.jsonl',
        'model_usage.sqlite3', 'annotations.sqlite3'))
    return {path: digest(path.read_bytes()) for path in paths if path.is_file()}


def reprocess_existing(run, *, document_ids):
    """Explicit IDs only; no URL guessing, downloads, retries, or manifest edits."""
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,80}', run) or not document_ids:
        raise ValueError('Require a safe run and explicit existing document IDs')
    runroot = (DATA / 'unified_runs' / run).resolve()
    if runroot.parent != (DATA / 'unified_runs').resolve() or (runroot / 'published').exists():
        raise ValueError('Cannot reprocess outside an unpublished unified run')
    implementations = _implementations()
    with writer_lock(run):
        documents, sources = validate_manifest_pair(runroot / 'supplement')
        by_id, by_url = {d['document_id']: d for d in documents}, {s['url']: s for s in sources}
        if len(set(document_ids)) != len(document_ids) or any(i not in by_id for i in document_ids):
            raise ValueError('Explicit reprocessing IDs must be known and unique')
        # Validate prior admissions first; a rerun never re-extracts or appends duplicates.
        apply_reprocessing_admissions(documents, sources, data=DATA, runroot=runroot)
        admission_path = runroot / 'supplement/reprocessing_admissions.jsonl'
        admitted = {r['document_id'] for r in read_jsonl(admission_path)}
        negative = {r['document_id'] for r in read_jsonl(runroot / 'supplement/admission_exclusions.jsonl')}
        protected = _protections(DATA, runroot)
        results = []
        for identifier in document_ids:
            original, source = by_id[identifier], by_url[by_id[identifier]['url']]
            if identifier in admitted:
                results.append(dict(document_id=identifier, status='already_admitted_not_duplicated'))
                continue
            if identifier in negative:
                raise ValueError('Negative audited admission cannot receive positive reprocessing')
            _original_pair(original, source)
            raw = _raw(DATA, original)
            cache = DATA / 'extracted' / (original['raw_sha256'] + extraction_suffix(original['source']))
            if cache.exists():
                raise ValueError('Existing source-bound extraction cache cannot be overwritten')
            print(f'BOEM local reparse: {identifier}; {len(raw)} preserved raw bytes', flush=True)
            try:
                parsed = parser_module.parse_boem_pdf(raw, source)
            except Exception as exc:
                result = dict(document_id=identifier, raw_sha256=original['raw_sha256'],
                    status='still_excluded', prior_error=_prior(original), error=str(exc),
                    implementations=implementations, model_only=True, human_gold=False,
                    semantic_approval=False, external_model_calls=0, human_labels_created=0)
                failures = runroot / 'supplement/reprocessing_failures.jsonl'
                if result not in read_jsonl(failures):
                    append_jsonl(failures, result)
                results.append(result)
                print(f'BOEM local reparse held: {identifier}; {exc}', flush=True)
                continue
            if parsed.get('published_date') != original.get('published_date') or _day(parsed['available_by']) < original['available_by']:
                raise ValueError('Reprocessing parser changed publication/backdated version')
            pages, removed = margin_module.remove_margin_furniture(parsed['pages'])
            artifacts = runroot / 'reports/boem-reprocessing' / (
                identifier + '-' + implementations['parser']['sha256'][:12] + '-' + implementations['script']['sha256'][:12])
            targets = [cache, artifacts / 'source_pages.json', artifacts / 'extraction_audit.json', artifacts / 'machine_proof.json']
            if any(path.exists() for path in targets):
                raise ValueError('Prior unregistered reprocessing artifacts exist; refuse overwrite or guessed recovery')
            pages_binding = _new_json(artifacts / 'source_pages.json', parsed['pages'])
            cache_binding = _new_json(cache, pages)
            audit = dict(raw_sha256=original['raw_sha256'], implementations=implementations,
                source_pdf_pages_sha256=canonical_sha(parsed['pages']), cache_sha256=cache_binding['sha256'],
                physical_pages=len(pages), pages_with_removed_margin_lines=removed,
                removal_policy='recognized_exact_repeated_margin_only; no body rewrites or synthetic facts')
            audit_binding = _new_json(artifacts / 'extraction_audit.json', audit)
            proof = dict(decision_version=VERSION, document_id=identifier, raw_sha256=original['raw_sha256'],
                original_document_sha256=canonical_sha(original), original_source_sha256=canonical_sha(source),
                implementations=implementations, source_pages=pages_binding, cache=cache_binding, audit=audit_binding,
                parsed_metadata={key: value for key, value in parsed.items() if key != 'pages'},
                model_only=True, human_gold=False, semantic_approval=False, accepted_passages=None,
                external_model_calls=0, human_labels_created=0)
            proof_binding = _new_json(artifacts / 'machine_proof.json', proof)
            updates = dict(available_by=parsed['available_by'], indexed_available_by=parsed['indexed_available_by'],
                body_date_verified=True, cover_month=parsed['cover_month'], pdf_creation_date=parsed['pdf_creation_date'],
                pdf_modified_date=parsed['pdf_modified_date'], physical_pages=len(pages),
                extraction_audit_path=audit_binding['path'], extraction_audit_sha256=audit_binding['sha256'],
                body_parser=parsed['parser'], body_parser_sha256=implementations['parser']['sha256'],
                body_word_count=sum(len(words(page)) for page in pages), publication_version_caution=parsed['version_caution'])
            derived = []
            for old in (original, source):
                value = copy.deepcopy(old)
                value.pop('ingest_exclusion_reason', None)
                value.pop('body_parser_error', None)
                value.update(updates, prior_error=_prior(old))
                derived.append(value)
            admission = dict(at=now(), decision_version=VERSION, document_id=identifier,
                raw_sha256=original['raw_sha256'], original_document_sha256=canonical_sha(original),
                original_source_sha256=canonical_sha(source), implementations=implementations,
                derived_document=derived[0], derived_source=derived[1], cache=cache_binding,
                audit=audit_binding, source_pages=pages_binding, proof=proof_binding,
                model_only=True, human_gold=False, semantic_approval=False)
            # The exact consumer gate validates prospective rows in memory.
            # No empty placeholder ledger, replacement, or recovery guesses.
            _validate_admission_rows(documents, sources, data=DATA, runroot=runroot,
                                      admissions=read_jsonl(admission_path) + [admission])
            _check_implementations(implementations)
            if any(not p.is_file() or digest(p.read_bytes()) != sha for p, sha in protected.items()):
                raise ValueError('Protected raw/cache/live/original manifests changed during local reprocessing')
            append_jsonl(admission_path, admission)
            results.append(dict(document_id=identifier, status='machine_version_checked_stage_only',
                raw_sha256=original['raw_sha256'], available_by=parsed['available_by'],
                published_date=parsed['published_date'], physical_pages=len(pages),
                cache=cache_binding, audit=audit_binding, proof=proof_binding))
            print(f'BOEM local repair admitted: {identifier}; {len(pages)} physical pages (not accepted passage count)', flush=True)
        apply_reprocessing_admissions(documents, sources, data=DATA, runroot=runroot)
        if any(not p.is_file() or digest(p.read_bytes()) != sha for p, sha in protected.items()):
            raise ValueError('Protected prior artifacts changed during local reprocessing')
        report = dict(at=now(), decision_version=VERSION, results=results, implementations=implementations,
            protected_prior_files_preserved=len(protected), original_manifest_objects_preserved=True,
            model_only=True, human_gold=False, semantic_approval=False, accepted_passages=None,
            external_model_calls=0, human_labels_created=0)
        receipt = runroot / 'reports/boem-reprocessing' / ('invocation-' + canonical_sha(report)[:20] + '.json')
        _new_json(receipt, report)
        return report


if __name__ == '__main__':
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument('--run', required=True)
    arguments.add_argument('--document-id', nargs='+', required=True)
    args = arguments.parse_args()
    print(json.dumps(reprocess_existing(args.run, document_ids=args.document_id), ensure_ascii=False, indent=2))
