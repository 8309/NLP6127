"""Compatibility entrypoint for the pure oil PDF chapter scope helper."""

from oil_evidence.pdf_scope import oil_page_numbers


def _inspect_samples():
    """Read cached samples, optionally fetching only the two observed URLs."""
    import io
    import json
    from pypdf import PdfReader
    from oil_evidence.corpus import Fetcher
    from oil_evidence.store import DATA, digest, now

    root = DATA / "expansion_runs/official-2018-2026-v2/discovery/oil-scope-samples"
    root.mkdir(parents=True, exist_ok=True)
    examples = [
        ("AU-REQ", "https://www.industry.gov.au/sites/default/files/2025-06/resources-and-energy-quarterly-june-2025.pdf"),
        ("UK-DUKES", "https://assets.publishing.service.gov.uk/media/68dbe477ef1c2f72bc1e4c4d/DUKES_2025_Chapters_1-7.pdf"),
    ]
    fetch = Fetcher()
    try:
        for source, url in examples:
            cached = list(root.glob(source + "-*.pdf"))
            if cached:
                path = cached[0]
                raw = path.read_bytes()
                final = url
            else:
                raw, final, _ = fetch.get(url)
                if not raw.startswith(b"%PDF") or len(raw) > 25 * 1024 * 1024:
                    raise ValueError("Invalid sample PDF")
                path = root / (source + "-" + digest(raw)[:16] + ".pdf")
                path.write_bytes(raw)
            pages = [page.extract_text() or '' for page in PdfReader(io.BytesIO(raw)).pages]
            print(json.dumps(dict(source=source, url=url, final_url=final, path=str(path),
                                  raw_sha256=digest(raw), bytes=len(raw), pages=len(pages), at=now(),
                                  oil_physical_pages=sorted(oil_page_numbers(pages, source)))), flush=True)
    finally:
        fetch.close()


if __name__ == "__main__":
    _inspect_samples()
