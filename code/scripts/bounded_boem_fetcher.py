"""Same official BOEM URLs, with a shared deadline for each complete fetch.

IPv4 binding is a local transport choice, not an alternate host or access-control
bypass. No threads continue downloading after a request returns. Native resolver
latency is still controlled by the operating system, not Python's read timeout.
"""
from __future__ import annotations

import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx

from discover_boem_buffer import official
from oil_evidence.corpus import Fetcher, USER_AGENT


class BoundedBoemFetcher(Fetcher):
    def __init__(self, max_elapsed_seconds=120, client=None):
        if not 1 <= max_elapsed_seconds <= 180:
            raise ValueError("Require a bounded 1..180 second per-fetch deadline")
        super().__init__()
        self.client.close()
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT}, follow_redirects=False,
            transport=httpx.HTTPTransport(local_address="0.0.0.0", retries=0),
            timeout=httpx.Timeout(15, connect=12),
        )
        self.max_elapsed_seconds = max_elapsed_seconds
        self.requested_urls = []

    @staticmethod
    def remaining(deadline):
        left = deadline - time.monotonic()
        if left <= 0:
            raise RuntimeError("BOEM per-fetch elapsed deadline reached; incomplete response is not ingested")
        return left

    def _wait(self, delay, deadline):
        if delay >= self.remaining(deadline):
            raise RuntimeError("BOEM per-fetch deadline cannot cover required rate/retry delay")
        if delay > 0:
            time.sleep(delay)

    def _request(self, url, deadline, *, robots=False):
        if not official(url):
            raise ValueError("Every BOEM request/robots redirect must retain an official HTTPS host")
        origin = f"https://{urlparse(url).netloc}"
        rp = self.robots.get(origin)
        delay = max(1, self.settings["request_delay_seconds"], rp.crawl_delay(USER_AGENT) or 0 if rp else 0)
        for attempt in range(3):
            self._wait(max(0, delay - (time.monotonic() - self.last.get(origin, 0))), deadline)
            self.last[origin] = time.monotonic()
            left = self.remaining(deadline)
            self.requested_urls.append(dict(url=url, robots=robots, attempt=attempt + 1))
            try:
                with self.client.stream("GET", url,
                        timeout=httpx.Timeout(min(15, left), connect=min(12, left))) as response:
                    if response.is_redirect:
                        return dict(redirect=urljoin(url, response.headers["location"]))
                    if robots and response.status_code in {404, 410}:
                        return dict(raw=b"", final=str(response.url), mime="", status=response.status_code)
                    response.raise_for_status()  # Never retry/bypass 403 or other HTTP failures.
                    cap = 512 * 1024 if robots else self.settings["max_document_mb"] * 1024**2
                    declared = response.headers.get("content-length", "")
                    if declared.isdigit() and int(declared) > cap:
                        raise ValueError("BOEM declared response size exceeds configured size limit")
                    chunks, size = [], 0
                    for chunk in response.iter_bytes():
                        size += len(chunk)
                        self.bytes += len(chunk)
                        self.remaining(deadline)
                        if size > cap:
                            raise ValueError("BOEM robots/document exceeds configured size limit")
                        if self.bytes > self.settings["download_budget_mb"] * 1024**2:
                            raise ValueError("BOEM download budget reached")
                        chunks.append(chunk)
                    self.remaining(deadline)
                    return dict(raw=b"".join(chunks), final=str(response.url),
                                mime=response.headers.get("content-type", ""), status=response.status_code)
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise
                self._wait(2**attempt, deadline)
        raise AssertionError("Unreachable bounded request state")

    def _ensure_robots(self, origin, deadline):
        if origin in self.robots:
            return
        url = origin + "/robots.txt"
        for _ in range(6):
            result = self._request(url, deadline, robots=True)
            if "redirect" in result:
                url = result["redirect"]
                if not official(url):
                    raise ValueError("BOEM robots redirect is not an official HTTPS URL")
                continue
            parser = RobotFileParser()
            if result["status"] in {404, 410}:
                parser.parse([])
            elif result["status"] == 200 and b"<html" not in result["raw"][:200].lower():
                parser.parse(result["raw"].decode("utf-8", errors="replace").splitlines())
            else:
                raise RuntimeError("Cannot verify BOEM robots.txt")
            self.robots[origin] = parser
            return
        raise ValueError("Too many BOEM robots redirects")

    def _get(self, url, deadline, depth):
        if depth > 5 or not official(url):
            raise ValueError("BOEM redirects must be bounded official HTTPS URLs")
        self.remaining(deadline)
        origin = f"https://{urlparse(url).netloc}"
        self._ensure_robots(origin, deadline)
        if not self.robots[origin].can_fetch(USER_AGENT, url):
            raise RuntimeError("robots.txt disallows " + url)
        result = self._request(url, deadline)
        if "redirect" in result:
            # Outside the request retry handler: target exhaustion cannot
            # restart the alias's retry budget.
            return self._get(result["redirect"], deadline, depth + 1)
        return result["raw"], result["final"], result["mime"]

    def get(self, url, _redirects=0):
        return self._get(url, time.monotonic() + self.max_elapsed_seconds, _redirects)
