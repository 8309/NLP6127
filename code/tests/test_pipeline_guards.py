import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from oil_evidence.corpus import Fetcher
from oil_evidence.store import digest


def test_robots_disallow_prevents_document_request(monkeypatch):
    calls = []
    def get(self, url, **kwargs):
        calls.append(url)
        return httpx.Response(200, request=httpx.Request("GET",url), text="User-agent: *\nDisallow: /private/")
    monkeypatch.setattr(httpx.Client, "get", get)
    with pytest.raises(RuntimeError, match="disallows"):
        Fetcher().get("https://example.org/private/report.pdf")
    assert calls == ["https://example.org/robots.txt"]


def test_robots_denied_fails_closed(monkeypatch):
    monkeypatch.setattr(httpx.Client, "get", lambda self,url,**kw: httpx.Response(403,request=httpx.Request("GET",url)))
    with pytest.raises(RuntimeError, match="verify robots"):
        Fetcher().get("https://example.org/report.pdf")


def test_redirect_target_rechecks_robots(monkeypatch):
    from contextlib import contextmanager
    calls=[]
    def robots(self,url,**kw):
        calls.append(url)
        text='User-agent: *\nDisallow: /private/' if 'destination' in url else 'User-agent: *\nDisallow:'
        return httpx.Response(200,request=httpx.Request('GET',url),text=text)
    @contextmanager
    def stream(self,method,url,**kw):
        calls.append(url)
        yield httpx.Response(302,request=httpx.Request('GET',url),headers={'location':'https://destination.test/private/report.pdf'})
    monkeypatch.setattr(httpx.Client,'get',robots)
    monkeypatch.setattr(httpx.Client,'stream',stream)
    monkeypatch.setattr('oil_evidence.corpus.time.sleep',lambda _:None)
    with pytest.raises(RuntimeError,match='disallows'):
        Fetcher().get('https://origin.test/report.pdf')
    assert calls==['https://origin.test/robots.txt','https://origin.test/report.pdf','https://destination.test/robots.txt']


def test_changed_corpus_rejects_old_dense_index(tmp_path, monkeypatch):
    import oil_evidence.retrieval as module
    monkeypatch.setattr(module, "DATA", tmp_path)
    p = dict(passage_id="a",text="crude oil supply",available_by="2025-01-01",regions=[])
    (tmp_path/"passages.jsonl").write_text(json.dumps(p)+"\n")
    (tmp_path/"index").mkdir()
    (tmp_path/"index/manifest.json").write_text(json.dumps(dict(corpus_sha256="stale",model="anything")))
    retriever = module.Retriever()
    assert retriever.search("oil",mode="bm25")
    with pytest.raises(ValueError, match="stale"):
        retriever.search("oil",mode="dense")


def test_formal_evaluation_rejects_unfrozen_pilot(tmp_path, monkeypatch):
    from oil_evidence import evaluation
    # This guard must work in a clean clone without the excluded real corpus.
    corpus = tmp_path / "synthetic-corpus"
    corpus.mkdir()
    (corpus / "passages.jsonl").write_text(
        json.dumps(dict(passage_id="fixture-only", text="Synthetic crude oil evidence.")) + "\n",
        encoding="utf-8")
    monkeypatch.setattr(evaluation, "DATA", corpus)
    f = tmp_path/"pilot.jsonl"
    f.write_text(json.dumps(dict(question="What is oil?",record_id="q"))+"\n")
    with pytest.raises(ValueError, match="frozen"):
        evaluation.run_experiments(f,smoke=False)
