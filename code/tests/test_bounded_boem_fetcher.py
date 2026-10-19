"""Offline adversarial checks; no requests reach BOEM or any other host."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import httpx
import pytest


ORIGIN = "https://www.boem.gov"
DOCUMENT = ORIGIN + "/document.pdf"
ALLOW = b"User-agent: *\nAllow: /\n"


class FakeClock:
    def __init__(self):
        self.value = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.value += seconds

    def advance(self, seconds):
        self.value += seconds


class TimedStream(httpx.SyncByteStream):
    """Each item is (elapsed time, received bytes or a transport exception)."""
    def __init__(self, clock, events):
        self.clock = clock
        self.events = events
        self.received_bytes = 0
        self.closed = False

    def __iter__(self):
        for elapsed, item in self.events:
            self.clock.advance(elapsed)
            if isinstance(item, Exception):
                raise item
            self.received_bytes += len(item)
            yield item

    def close(self):
        self.closed = True


@pytest.fixture
def harness(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("bounded_boem_test_subject", scripts / "bounded_boem_fetcher.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = FakeClock()
    # Replace only this module's time binding, not pytest/httpx/global time.
    monkeypatch.setattr(module, "time", clock)
    fetchers = []

    def make(handler, *, seconds=120, settings=None):
        requests = []

        def wrapped(request):
            requests.append((str(request.url), clock.monotonic(), dict(request.extensions.get("timeout", {}))))
            return handler(request)

        client = httpx.Client(transport=httpx.MockTransport(wrapped), follow_redirects=False)
        fetch = module.BoundedBoemFetcher(max_elapsed_seconds=seconds, client=client)
        fetch.settings = dict(fetch.settings)
        fetch.settings.update(request_delay_seconds=1, max_document_mb=25, download_budget_mb=300)
        fetch.settings.update(settings or {})
        fetchers.append(fetch)
        return fetch, requests

    yield module, clock, make
    for fetch in fetchers:
        fetch.close()


def allow_or(request, response):
    return httpx.Response(200, content=ALLOW) if request.url.path == "/robots.txt" else response


def test_success_returns_only_complete_payload_and_accounts_robots(harness):
    _, clock, make = harness
    stream = TimedStream(clock, [(0.2, b"first"), (0.2, b"second")])
    fetch, requests = make(lambda r: allow_or(r, httpx.Response(200, stream=stream, headers={"content-type": "application/pdf"})))
    assert fetch.get(DOCUMENT) == (b"firstsecond", DOCUMENT, "application/pdf")
    assert fetch.bytes == len(ALLOW) + len(b"firstsecond")
    assert stream.closed
    assert [url for url, _, _ in requests] == [ORIGIN + "/robots.txt", DOCUMENT]
    assert requests[1][1] - requests[0][1] >= 1


@pytest.mark.parametrize("seconds", [0, 0.9, 181])
def test_invalid_deadline_rejected_before_any_network(harness, seconds):
    module, _, _ = harness
    with pytest.raises(ValueError, match="1..180"):
        module.BoundedBoemFetcher(max_elapsed_seconds=seconds)


@pytest.mark.parametrize("url", [
    "http://www.boem.gov/file.pdf", "https://www.boem.gov.evil.test/file.pdf",
    "https://evil.test/file.pdf", "https://user@www.boem.gov/file.pdf", "https://www.boem.gov:444/file.pdf",
])
def test_untrusted_start_url_never_reaches_transport(harness, url):
    _, _, make = harness
    fetch, requests = make(lambda r: pytest.fail("Untrusted URL reached transport"))
    with pytest.raises(ValueError, match="official HTTPS"):
        fetch.get(url)
    assert requests == []


@pytest.mark.parametrize("location", [
    "http://www.boem.gov/robots.txt", "https://evil.test/robots.txt",
    "https://www.boem.gov.evil.test/robots.txt", "https://user@www.boem.gov/robots.txt",
])
def test_robots_redirect_cannot_escape_official_https(harness, location):
    _, _, make = harness
    fetch, requests = make(lambda r: httpx.Response(302, headers={"location": location}))
    with pytest.raises(ValueError, match="robots redirect.*official HTTPS"):
        fetch.get(DOCUMENT)
    assert [r[0] for r in requests] == [ORIGIN + "/robots.txt"]


def test_robots_redirect_loop_is_bounded(harness):
    _, _, make = harness
    fetch, requests = make(lambda r: httpx.Response(302, headers={"location": "/robots.txt"}))
    with pytest.raises(ValueError, match="Too many BOEM robots redirects"):
        fetch.get(DOCUMENT)
    assert len(requests) == 6
    assert all(item["robots"] for item in fetch.requested_urls)


def test_robots_disallow_never_requests_document(harness):
    _, _, make = harness
    fetch, requests = make(lambda r: httpx.Response(200, content=b"User-agent: *\nDisallow: /document.pdf\n"))
    with pytest.raises(RuntimeError, match="robots.txt disallows"):
        fetch.get(DOCUMENT)
    assert [r[0] for r in requests] == [ORIGIN + "/robots.txt"]


@pytest.mark.parametrize("status", [404, 410])
def test_robots_confirmed_absent_allows_document(harness, status):
    _, _, make = harness
    def handler(request):
        return httpx.Response(status) if request.url.path == "/robots.txt" else httpx.Response(200, content=b"PDF")
    fetch, _ = make(handler)
    assert fetch.get(DOCUMENT)[0] == b"PDF"


def test_robots_html_refuses_document(harness):
    _, _, make = harness
    fetch, requests = make(lambda r: httpx.Response(200, content=b"<html>access page</html>"))
    with pytest.raises(RuntimeError, match="Cannot verify BOEM robots.txt"):
        fetch.get(DOCUMENT)
    assert len(requests) == 1


def test_robot_crawl_delay_and_cached_policy_apply_to_all_document_requests(harness):
    _, _, make = harness
    policy = b"User-agent: *\nAllow: /\nCrawl-delay: 3\n"
    def handler(request):
        return httpx.Response(200, content=policy if request.url.path == "/robots.txt" else b"PDF")
    fetch, requests = make(handler)
    fetch.get(DOCUMENT)
    fetch.get(ORIGIN + "/other.pdf")
    assert [url for url, _, _ in requests].count(ORIGIN + "/robots.txt") == 1
    assert [requests[i + 1][1] - requests[i][1] for i in range(2)] == [3, 3]


def test_configured_delay_cannot_weaken_one_second_minimum(harness):
    _, _, make = harness
    fetch, requests = make(lambda r: allow_or(r, httpx.Response(200, content=b"PDF")), settings={"request_delay_seconds": 0})
    fetch.get(DOCUMENT)
    assert requests[1][1] - requests[0][1] == 1


def test_document_cap_closes_response_without_returning_partial_payload(harness):
    _, clock, make = harness
    stream = TimedStream(clock, [(0, b"1234"), (0, b"56789"), (0, b"not-read")])
    fetch, _ = make(lambda r: allow_or(r, httpx.Response(200, stream=stream)), settings={"max_document_mb": 8 / 1024**2})
    with pytest.raises(ValueError, match="size limit"):
        fetch.get(DOCUMENT)
    assert stream.closed and stream.received_bytes == 9
    assert fetch.bytes == len(ALLOW) + 9


@pytest.mark.parametrize("robots", [False, True])
def test_declared_oversize_rejected_without_consuming_body(harness, robots):
    _, clock, make = harness
    stream = TimedStream(clock, [(20, b"not-read")])
    declared = 512 * 1024 + 1 if robots else 9
    def handler(request):
        if not robots and request.url.path == "/robots.txt":
            return httpx.Response(200, content=ALLOW)
        return httpx.Response(200, stream=stream, headers={"content-length": str(declared)})
    fetch, requests = make(handler, settings={"max_document_mb": 8 / 1024**2})
    with pytest.raises(ValueError, match="declared response size.*size limit"):
        fetch.get(DOCUMENT)
    assert stream.closed and stream.received_bytes == 0
    assert fetch.bytes == (0 if robots else len(ALLOW))
    assert len(requests) == (1 if robots else 2)


@pytest.mark.parametrize("declared", ["4", "invalid", "-20"])
def test_underreported_or_invalid_length_cannot_bypass_actual_body_cap(harness, declared):
    _, clock, make = harness
    stream = TimedStream(clock, [(0, b"123456789")])
    fetch, _ = make(lambda r: allow_or(r, httpx.Response(200, stream=stream, headers={"content-length": declared})),
                    settings={"max_document_mb": 8 / 1024**2})
    with pytest.raises(ValueError, match="size limit"):
        fetch.get(DOCUMENT)
    assert stream.closed and fetch.bytes == len(ALLOW) + 9


def test_robots_cap_is_independent_and_charged(harness):
    _, clock, make = harness
    stream = TimedStream(clock, [(0, b"x" * (512 * 1024 + 1))])
    fetch, requests = make(lambda r: httpx.Response(200, stream=stream))
    with pytest.raises(ValueError, match="size limit"):
        fetch.get(DOCUMENT)
    assert fetch.bytes == 512 * 1024 + 1
    assert stream.closed and len(requests) == 1


def test_download_budget_includes_robots_and_failed_document_chunks(harness):
    _, clock, make = harness
    stream = TimedStream(clock, [(0, b"abc"), (0, b"def"), (0, b"not-read")])
    fetch, _ = make(lambda r: allow_or(r, httpx.Response(200, stream=stream)), settings={"download_budget_mb": (len(ALLOW) + 5) / 1024**2})
    with pytest.raises(ValueError, match="download budget"):
        fetch.get(DOCUMENT)
    assert fetch.bytes == len(ALLOW) + 6
    assert stream.closed and stream.received_bytes == 6


def test_exhausted_budget_during_robots_never_requests_document(harness):
    _, _, make = harness
    fetch, requests = make(lambda r: httpx.Response(200, content=ALLOW), settings={"download_budget_mb": (len(ALLOW) - 1) / 1024**2})
    with pytest.raises(ValueError, match="download budget"):
        fetch.get(DOCUMENT)
    assert len(requests) == 1 and fetch.bytes == len(ALLOW)


def test_slow_chunk_deadline_is_not_a_retry_and_closes_partial_response(harness):
    _, clock, make = harness
    stream = TimedStream(clock, [(1, b"first"), (3, b"late"), (0, b"not-read")])
    fetch, requests = make(lambda r: allow_or(r, httpx.Response(200, stream=stream)), seconds=5)
    with pytest.raises(RuntimeError, match="elapsed deadline.*incomplete response"):
        fetch.get(DOCUMENT)
    assert stream.closed and stream.received_bytes == len(b"firstlate")
    assert [r[0] for r in requests].count(DOCUMENT) == 1


def test_received_chunk_at_deadline_is_still_charged_to_download_budget(harness):
    _, clock, make = harness
    stream = TimedStream(clock, [(1, b"first"), (3, b"late")])
    fetch, _ = make(lambda r: allow_or(r, httpx.Response(200, stream=stream)), seconds=5)
    with pytest.raises(RuntimeError, match="elapsed deadline"):
        fetch.get(DOCUMENT)
    # Even a refused partial response consumed these bytes at the transport.
    assert fetch.bytes == len(ALLOW) + stream.received_bytes


def test_empty_response_over_deadline_is_not_returned(harness):
    _, clock, make = harness
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, content=ALLOW)
        clock.advance(5)
        return httpx.Response(200, content=b"")
    fetch, _ = make(handler, seconds=5)
    with pytest.raises(RuntimeError, match="elapsed deadline"):
        fetch.get(DOCUMENT)


def test_required_rate_delay_cannot_run_past_deadline(harness):
    _, clock, make = harness
    def handler(request):
        clock.advance(0.5)
        return httpx.Response(200, content=ALLOW)
    fetch, requests = make(handler, seconds=1)
    with pytest.raises(RuntimeError, match="required rate/retry delay"):
        fetch.get(DOCUMENT)
    assert len(requests) == 1 and clock.monotonic() <= 1001


def test_robots_body_and_target_share_deadline_and_timeout_budget(harness):
    _, clock, make = harness
    robots = TimedStream(clock, [(3, ALLOW)])
    target = TimedStream(clock, [(2, b"late")])
    def handler(request):
        return httpx.Response(200, stream=robots if request.url.path == "/robots.txt" else target)
    fetch, requests = make(handler, seconds=5)
    with pytest.raises(RuntimeError, match="elapsed deadline"):
        fetch.get(DOCUMENT)
    assert requests[1][2]["read"] == 2
    assert requests[1][2]["connect"] == 2
    assert target.closed and robots.closed


def test_target_retry_exhaustion_never_restarts_redirect_alias(harness):
    _, _, make = harness
    alias = ORIGIN + "/alias"
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, content=ALLOW)
        if str(request.url) == alias:
            return httpx.Response(302, headers={"location": "/document.pdf"})
        raise httpx.ReadTimeout("fixture transient read failure", request=request)
    fetch, requests = make(handler)
    with pytest.raises(httpx.ReadTimeout):
        fetch.get(alias)
    urls = [r[0] for r in requests]
    assert urls.count(alias) == 1 and urls.count(DOCUMENT) == 3
    assert [r["attempt"] for r in fetch.requested_urls if r["url"] == DOCUMENT] == [1, 2, 3]


def test_robots_transient_failures_have_three_attempt_limit(harness):
    _, _, make = harness
    def handler(request):
        raise httpx.ConnectError("fixture unreachable", request=request)
    fetch, requests = make(handler)
    with pytest.raises(httpx.ConnectError):
        fetch.get(DOCUMENT)
    assert [r[0] for r in requests] == [ORIGIN + "/robots.txt"] * 3
    assert [r["attempt"] for r in fetch.requested_urls] == [1, 2, 3]


def test_retry_discards_earlier_partial_payload_but_counts_all_received_bytes(harness):
    _, clock, make = harness
    first = TimedStream(clock, [(0, b"partial"), (0, httpx.ReadTimeout("fixture incomplete stream"))])
    final = TimedStream(clock, [(0, b"complete")])
    document_calls = 0
    def handler(request):
        nonlocal document_calls
        if request.url.path == "/robots.txt":
            return httpx.Response(200, content=ALLOW)
        document_calls += 1
        return httpx.Response(200, stream=first if document_calls == 1 else final)
    fetch, _ = make(handler)
    assert fetch.get(DOCUMENT)[0] == b"complete"
    assert fetch.bytes == len(ALLOW) + len(b"partialcomplete")
    assert first.closed and final.closed


def test_retry_delay_respects_shared_deadline(harness):
    _, clock, make = harness
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, content=ALLOW)
        clock.advance(3.5)
        raise httpx.ReadTimeout("fixture failure", request=request)
    fetch, requests = make(handler, seconds=5)
    with pytest.raises(RuntimeError, match="required rate/retry delay"):
        fetch.get(DOCUMENT)
    assert [r[0] for r in requests].count(DOCUMENT) == 1
    assert clock.monotonic() <= 1005


@pytest.mark.parametrize("robots", [False, True])
def test_403_is_requested_once_without_retry_or_bypass(harness, robots):
    _, _, make = harness
    def handler(request):
        if not robots and request.url.path == "/robots.txt":
            return httpx.Response(200, content=ALLOW)
        return httpx.Response(403)
    fetch, requests = make(handler)
    with pytest.raises(httpx.HTTPStatusError) as err:
        fetch.get(DOCUMENT)
    assert err.value.response.status_code == 403
    path = ORIGIN + "/robots.txt" if robots else DOCUMENT
    assert [r[0] for r in requests].count(path) == 1
    assert len(requests) == (1 if robots else 2)


def test_redirect_document_rechecks_destination_robots(harness):
    _, _, make = harness
    destination = "https://boem.gov/document.pdf"
    def handler(request):
        if str(request.url) == ORIGIN + "/robots.txt":
            return httpx.Response(200, content=ALLOW)
        if str(request.url) == DOCUMENT:
            return httpx.Response(302, headers={"location": destination})
        if str(request.url) == "https://boem.gov/robots.txt":
            return httpx.Response(200, content=b"User-agent: *\nDisallow: /\n")
        pytest.fail("Disallowed redirected document was requested")
    fetch, requests = make(handler)
    with pytest.raises(RuntimeError, match="robots.txt disallows"):
        fetch.get(DOCUMENT)
    assert [r[0] for r in requests] == [ORIGIN + "/robots.txt", DOCUMENT, "https://boem.gov/robots.txt"]


def test_document_redirect_loop_is_bounded(harness):
    _, _, make = harness
    fetch, requests = make(lambda r: allow_or(r, httpx.Response(302, headers={"location": "/document.pdf"})))
    with pytest.raises(ValueError, match="redirects must be bounded"):
        fetch.get(DOCUMENT)
    assert [r[0] for r in requests].count(DOCUMENT) == 6


def test_redirect_consumes_same_deadline_instead_of_restarting_it(harness):
    _, clock, make = harness
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, content=ALLOW)
        clock.advance(1.5)
        return httpx.Response(302, headers={"location": "/next.pdf"})
    fetch, requests = make(handler, seconds=4)
    with pytest.raises(RuntimeError, match="elapsed deadline"):
        fetch.get(DOCUMENT)
    assert [r[0] for r in requests] == [ORIGIN + "/robots.txt", DOCUMENT, ORIGIN + "/next.pdf"]
