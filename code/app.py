from datetime import date
from html import escape
import json
import os
import time
from pathlib import Path

import streamlit as st

from oil_evidence.annotations import agreement, validate_record
from oil_evidence.corpus_versions import annotation_passages, active_label_coverage, original_parent
from oil_evidence.query import query
from oil_evidence.codex_runtime import load_staged_runtime, query_codex
from oil_evidence.retrieval import Retriever
from oil_evidence.store import DATA, config, digest, latest_reviews, read_jsonl, save_review, mark_suggestion_view, has_suggestion_view

st.set_page_config(page_title="Oil Evidence", page_icon="🛢", layout="wide")
st.title("Oil Evidence · Crude Oil Evidence Workbench")
staged_root = os.environ.get("OIL_CANONICAL_ROOT")
staged_run = os.environ.get("OIL_STAGED_RUN", "semantic-page-v2-buffer")
provider_options = ["codex", "deepseek"]
provider_default = os.environ.get("OIL_GENERATION_PROVIDER", "codex")
provider = st.sidebar.selectbox("Generation provider", provider_options,
    index=provider_options.index(provider_default) if provider_default in provider_options else 0,
    format_func=lambda value: "Codex · Existing ChatGPT login" if value == "codex" else "DeepSeek · Optional API")
with st.sidebar.expander("Model connection · Local process"):
    if provider == "codex":
        st.caption("No API key is needed. Native Codex uses the existing local ChatGPT login; source text is supplied as data and model tools are disabled.")
        st.caption("Subscription credits and dollar cost are not measured. Native token usage and actual elapsed time are retained for each invocation.")
    else:
        st.caption("DeepSeek remains optional, with the existing cumulative estimated USD 5 budget. Keys are not saved to files.")
        with st.form("runtime_key"):
            runtime_key = st.text_input("DeepSeek API key", type="password")
            use_key = st.form_submit_button("Use only in the current server process")
        if use_key:
            if runtime_key.strip():
                os.environ[config()["llm"]["api_key_env"]] = runtime_key.strip()
                st.success("Configured for this process; the key expires when the server stops.")
            else:
                st.error("The key is empty.")
        st.caption("Connection status: " + ("A key is set for this process" if os.environ.get(config()["llm"]["api_key_env"]) else "No key is set; source preview remains available"))
        if staged_root:
            st.warning("The staged Codex workbench does not invoke DeepSeek or create a new budget ledger. Use the original configured workbench for that optional path.")
if staged_root:
    st.info("Read-only staged library: " + staged_run + ". The original default library and archived labels are not switched or rewritten.")
search_tab, review_tab, passage_tab, status_tab = st.tabs(["Evidence search", "Annotation review", "Passage labels", "Data and progress"])

# Existing annotations always use their frozen text version, never child IDs.
if staged_root:
    annotation_rows, annotation_map = [], {}
else:
    try:
        annotation_rows = annotation_passages()
        annotation_map = {p["passage_id"]: p for p in annotation_rows}
    except ValueError as exc:
        st.error("Annotation corpus version validation failed: " + str(exc))
        st.stop()


@st.cache_resource
def load_retriever(corpus_mtime, index_mtime):
    return Retriever()


@st.cache_resource
def load_staged(root, run, receipt_mtime):
    return load_staged_runtime(root, run)


def get_staged_runtime():
    if not staged_root:
        raise ValueError("Start with scripts/start_codex_workbench.py --canonical-root PATH to select a verified read-only stage.")
    receipt = Path(staged_root) / "data/unified_runs" / staged_run / "reports/index_build.json"
    return load_staged(staged_root, staged_run, receipt.stat().st_mtime_ns)


def get_retriever():
    if staged_root:
        return get_staged_runtime().retriever
    path = DATA / "passages.jsonl"
    index = DATA / "index/manifest.json"
    return load_retriever(path.stat().st_mtime_ns, index.stat().st_mtime_ns if index.exists() else 0)


with search_tab:
    st.caption("English evidence corpus. Retrieval quality for Chinese queries has not been validated with the local English embedding model; use English questions for formal evaluation.")
    with st.form("query"):
        question = st.text_area("Research question", "What factors affect global crude oil supply and inventories?")
        a, b, c = st.columns(3)
        cutoff = a.date_input("Source cutoff date", value=date.fromisoformat(config()["corpus"]["end_date"]))
        region = b.selectbox("Region (source-text keyword filter)", ["All", "Middle East", "Europe", "Asia", "Americas", "Africa"])
        mode = c.selectbox("Retrieval method", ["hybrid", "bm25", "dense"])
        temporal = st.checkbox("A: Use only sources available by the cutoff date", value=True)
        verify = st.checkbox("B: Check citation support (source preview checks verbatim quotes only)", value=False)
        answer_modes = ["llm", "extractive"] if staged_root and provider == "codex" else ["extractive", "llm"]
        generation = st.selectbox("Answer mode", answer_modes, format_func=lambda value:
            "Source excerpt preview · No model call" if value == "extractive" else
            "Codex-generated answer · No API key needed" if provider == "codex" else "DeepSeek-generated answer · Configured API required")
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        try:
            with st.spinner("Retrieving evidence and preparing the answer…"):
                if staged_root:
                    if generation == "llm" and provider != "codex":
                        raise ValueError("This read-only staged workbench uses Codex. DeepSeek requires the original workbench and its retained cost ledger.")
                    st.session_state["query_result"] = query_codex(question, get_staged_runtime(), mode=mode,
                        as_of=cutoff.isoformat(), region=None if region == "All" else region,
                        temporal=temporal, verify=verify, generation="codex" if generation == "llm" else "extractive")
                else:
                    if generation == "llm" and provider == "codex":
                        raise ValueError("Codex generation needs an explicit verified stage. Start scripts/start_codex_workbench.py with the canonical data root.")
                    st.session_state["query_result"] = query(question, mode=mode, as_of=cutoff.isoformat(),
                        region=None if region == "All" else region, temporal=temporal, verify=verify,
                        generation=generation, retriever=get_retriever())
        except Exception as exc:
            st.error(str(exc))
    result = st.session_state.get("query_result")
    if result:
        st.caption(f"{result['retrieval_mode']} · {result['total_seconds']:.2f} seconds · {result['generation']}")
        if result["generation"] == "codex":
            st.caption(f"Retrieval and input verification: {result['retrieval_seconds']:.3f} s · Native generation: {result['generation_seconds']:.3f} s · Actual single-query end to end: {result['total_seconds']:.3f} s")
            st.caption("Evidence sufficiency (machine): " + str(result.get("answerability")) + "; exact quotes and citations are checked, but this alone does not verify semantic faithfulness.")
            with st.expander("Codex execution usage and audit records"):
                st.json([dict(execution_id=call["execution_id"], observed_model=call["observed_model"],
                              usage=call["usage"], generation_seconds=call["generation_seconds"],
                              receipt_directory=call["artifact_directory"]) for call in result["model_calls"]])
        for limitation in result["limitations"]:
            st.info(limitation)
        for claim in result["claims"]:
            st.write(claim["text"])
            st.caption("Evidence: " + ", ".join(claim["citations"]))
        if result.get("quotes"):
            with st.expander("Exact source quotes supplied by Codex"):
                for quote in result["quotes"]:
                    st.write(quote["text"])
                    st.caption(quote["passage_id"])
        for p in result["evidence"]:
            with st.expander(f"#{p['rank']} {p['title']} · p.{p['page']} · {p['available_by']}"):
                st.write(p["text"])
                st.markdown(f"[Open original source]({p['url']})")
                st.caption(f"{p['passage_id']} · Date precision: {p['date_precision']} · Score {p['score']:.4f}")
                if p.get("parent_passage_id") or p.get("source_page_id"):
                    try:
                        parent = get_staged_runtime().pages[p["source_page_id"]] if staged_root else original_parent(p)
                        is_page = bool(p.get("source_page_id"))
                        label = "Cleaned source page" if is_page else "Original parent passage"
                        context_id = parent["source_page_id"] if is_page else parent["passage_id"]
                        with st.expander(label + " · Context only, not an additional retrieval passage"):
                            st.caption("This context is not automatically cited in the answer or counted in retrieval rankings or corpus passage totals.")
                            st.write(parent["text"])
                            st.caption(f"{label} {context_id} · p.{parent['page']} · {parent['available_by']}")
                    except ValueError as exc:
                        st.warning("Source context version validation failed; context is not displayed: " + str(exc))
        st.download_button("Export query JSON", json.dumps(result, ensure_ascii=False, indent=2), file_name="query.json", mime="application/json")

with review_tab:
    if staged_root:
        st.info("Historical annotation archives stay in the original workbench. The new staged passages do not inherit old labels.")
    st.caption("Codex review results are machine annotations, not human reviews. The human review form below records a judgment only when you save it.")
    if annotation_rows != read_jsonl(DATA / "passages.jsonl"):
        st.info("Archived annotations: question-answer candidate pools and citations refer to the original frozen corpus; they are not automatically migrated to the current resegmented retrieval passages.")
    datasets = (["codex"] if (DATA / "annotations/codex.jsonl").exists() else []) + ["pilot", "candidates"]
    dataset = st.selectbox("Review dataset", datasets, format_func=lambda x: {
        "codex": "Codex completed · 100 records", "pilot": "Original annotation queue · 50 questions", "candidates": "Original Flash candidates · 50 records"}[x])
    records = read_jsonl(DATA / f"annotations/{dataset}.jsonl")
    if dataset == "codex":
        outcomes = {name: sum(r["codex_review"]["result"] == name for r in records)
                    for name in ["passed", "corrected", "partial", "insufficient"]}
        st.success(f"Completed {len(records)} records: No obvious errors {outcomes['passed']} · Corrected {outcomes['corrected']} · Partially answerable {outcomes['partial']} · Insufficient evidence {outcomes['insufficient']}")
        view = st.selectbox("Review result filter", ["All", "Corrected", "Partially answerable / Insufficient evidence", "Not recommended for formal evaluation"])
        if view == "Corrected":
            records = [r for r in records if r["codex_review"]["result"] == "corrected"]
        elif view == "Partially answerable / Insufficient evidence":
            records = [r for r in records if r["answerability"] != "sufficient"]
        elif view == "Not recommended for formal evaluation":
            records = [r for r in records if not r["recommended_for_oil_evaluation"]]
    if not records:
        st.info("No review queue is available. Run oil pilot; after configuring a model, you can run oil generate.")
    else:
        who = st.text_input("Human annotator ID (use distinct IDs for distinct people)", key="annotator")
        reviews = latest_reviews()
        completed = {r["record_id"] for r in reviews if r["annotator"] == who}
        st.caption(f"The current annotator has saved {len(completed)} records; all records retain an append-only revision history.")
        index = st.selectbox("Record", range(len(records)), format_func=lambda i: f"{'✓ ' if records[i]['record_id'] in completed else ''}{records[i]['record_id']}: {records[i]['question'][:85]}")
        record = records[index]
        key = record["record_id"]
        view_key = f"{dataset}:{key}:{who}"
        st.session_state.setdefault("start:" + view_key, time.monotonic())
        # Keep the full question above potentially long machine suggestions.
        # Escape source text so the responsive display cannot inject HTML.
        with st.container(border=True):
            st.markdown("### Full question")
            st.markdown(
                '<div aria-label="Full question" style="white-space:pre-wrap;overflow-wrap:anywhere;'
                'font-size:1.15rem;line-height:1.65;">'
                + escape(record["question"]) + '</div>',
                unsafe_allow_html=True,
            )
            if record.get("question_zh"):
                st.write(record["question_zh"])
            st.caption("Cutoff date: " + str(record.get("as_of")))
        shown = st.checkbox("Show model/candidate answer suggestions (recorded as assisted review)", value=dataset == "codex", key="show:" + view_key)
        if shown:
            st.session_state["exposed:" + view_key] = True
            mark_suggestion_view(key, who)
            review = record.get("codex_review")
            if review:
                result_label = {"passed": "No obvious errors", "corrected": "Corrected", "partial": "Partially answerable", "insufficient": "Insufficient evidence"}[review["result"]]
                st.subheader("Codex review result · " + result_label)
                st.write(review["note_zh"])
                if not record["recommended_for_oil_evaluation"]:
                    st.warning("Machine annotation is complete, but scope, duplicate-template, or source-date issues make this record unsuitable for the formal oil evaluation set at present.")
            st.caption("Evidence sufficiency: " + {"sufficient": "Sufficient", "partial": "Partially answerable", "insufficient": "Insufficient", "unlabelled": "Not yet labelled"}.get(record.get("answerability"), "Not yet labelled")
                       + " · Event status: " + {"announced": "Announced", "effective": "Effective", "observed": "Reported as having occurred", "recovered": "Recovered", "uncertain": "Forecast / Mixed status / Uncertain", "not_applicable": "Not applicable", "unlabelled": "Not yet labelled"}.get(record.get("event_status"), "Not yet labelled"))
            st.markdown("#### Reference answer points (machine)")
            for n, point in enumerate(record.get("answer_points") or [], 1):
                st.write(f"{n}. {point}")
            if not record.get("answer_points"):
                st.info("These original records have no machine answers; select the completed Codex dataset to view the new annotations.")
            with st.expander("Verbatim quotes supporting the answer (machine)"):
                for q in record.get("quotes") or []:
                    st.write(q["text"])
                    st.caption(q["passage_id"])
                if not record.get("quotes"):
                    st.caption("No quotes sufficiently support a substantive answer; evidence is not fabricated.")
            with st.expander("View original annotation JSON"):
                st.json({k: record.get(k) for k in ["answer_points", "quotes", "answerability", "event_status", "codex_review", "original_annotation", "model_verification"]})
        passage_map = annotation_map
        pool = record.get("candidate_pool_ids", record.get("evidence_ids", []))
        if record.get("validation_errors"):
            st.warning("Machine pre-validation found issues: " + "; ".join(record["validation_errors"]))
        prior = next((r for r in reviews if r["annotator"] == who and r["record_id"] == key), None)
        previous = prior["payload"] if prior else {}
        qrels = {}
        quotes = []
        for pid in pool:
            p = passage_map.get(pid)
            if not p:
                st.error(f"Source passage missing: {pid}")
                continue
            with st.expander(f"{p['title']} / p.{p['page']} / {pid}", expanded=True):
                st.write(p["text"])
                st.markdown(f"[Original source]({p['url']}) · Available by {p['available_by']}")
                if shown and pid in record.get("qrels", {}):
                    st.caption("Machine relevance: " + {0: "Not relevant", 1: "Background / Partially relevant", 2: "Direct answer support"}[record["qrels"][pid]] + "; the human judgment below is not automatically populated with machine labels.")
                grade = st.selectbox("Relevance", [-1, 0, 1, 2],
                    index=[-1, 0, 1, 2].index(previous.get("qrels", {}).get(pid, -1)),
                    format_func=lambda x: {-1: "Not judged", 0: "Not relevant", 1: "Background / Partially relevant", 2: "Direct answer support"}[x], key=f"grade:{view_key}:{pid}")
                if grade >= 0:
                    qrels[pid] = grade
                old_quote = next((q["text"] for q in previous.get("quotes", []) if q["passage_id"] == pid), "")
                quote = st.text_area("Supporting source excerpt (copy verbatim; leave blank to omit this passage)", value=old_quote, key=f"quote:{view_key}:{pid}")
                if quote.strip():
                    quotes.append(dict(passage_id=pid, text=quote.strip()))
        with st.form("review:" + view_key):
            edited_question = st.text_area("Final question", value=previous.get("question", record["question"]))
            options = ["unlabelled", "sufficient", "partial", "insufficient"]
            answerability = st.selectbox("Evidence sufficiency", options, index=options.index(previous.get("answerability", "unlabelled")))
            statuses = ["unlabelled", "announced", "effective", "observed", "recovered", "uncertain", "not_applicable"]
            event_status = st.selectbox("Event status", statuses, index=statuses.index(previous.get("event_status", "unlabelled")))
            points = st.text_area("Reference answer points: one per line", value="\n".join(previous.get("answer_points", [])))
            group = st.text_input("Event group ID (the same event and paraphrased questions must share a group)", value=previous.get("event_group", record["event_group"]))
            group_checked = st.checkbox("I verified the event group, including the same event across documents", value=not previous.get("event_group_review_required", True))
            accept = st.checkbox("Accept as a candidate formal evaluation record", value=previous.get("status") == "accepted")
            note = st.text_area("Notes / Error explanation", value=previous.get("note", ""))
            save = st.form_submit_button("Save human judgment")
        if save:
            payload = dict(question=edited_question.strip(), answerability=answerability, event_status=event_status,
                           answer_points=[p.strip() for p in points.splitlines() if p.strip()], event_group=group,
                           event_group_review_required=not group_checked, qrels=qrels, quotes=quotes,
                           evidence_ids=[q["passage_id"] for q in quotes], note=note,
                           status="accepted" if accept else "draft", as_of=record.get("as_of"),
                           elapsed_seconds=time.monotonic() - st.session_state["start:" + view_key])
            errors = validate_record(payload, passage_map) if accept else []
            if accept and (not group_checked or not qrels):
                errors.append("Formal records require a verified event group and completed relevance judgments")
            exposed = st.session_state.get("exposed:" + view_key, False) or has_suggestion_view(key, who) or (prior and prior["mode"] == "human_reviewed")
            if errors:
                st.error("; ".join(errors))
            else:
                try:
                    save_review(key, who, "human_reviewed" if exposed else "human_independent", payload)
                    st.success("Saved. Model-generated questions remain recorded as synthetic questions, not human-original questions.")
                except ValueError as exc:
                    st.error(str(exc))

with passage_tab:
    progress_path = DATA / "reports/passage_annotation_progress.json"
    if not progress_path.exists():
        st.info("Source passage annotation batches have not been prepared.")
    else:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        current_passages = read_jsonl(DATA / "passages.jsonl")
        annotation_archive = len(current_passages) != progress['total'] or digest((DATA / 'passages.jsonl').read_bytes()) != progress['corpus_sha256']
        if annotation_archive:
            st.info(f"Archived annotations: the current retrieval corpus has {len(current_passages):,} passages; the {progress['validated']:,} machine labels below apply only to the frozen source text. Resegmented passages do not automatically inherit labels, and question-answer candidate pools are not ground truth for the current full corpus.")
        st.subheader(f"{'Archived ' if annotation_archive else ''}source passage annotations · {progress['validated']} / {progress['total']}")
        st.progress(progress["validated"] / progress["total"])
        st.caption(f"{progress['remaining']} passages remain; these are passage-level topic and event labels, not question-answer annotations. Machine labels do not count as human judgments.")
        second_progress_path = DATA / "reports/passage_review_progress.json"
        if second_progress_path.exists():
            second = json.loads(second_progress_path.read_text(encoding="utf-8"))
            st.caption(f"Second-pass machine review: blind annotations {second['blind_completed']} / {second['total']}; comparison and adjudication {second['reviewed']} / {second['total']}. Prepared inputs do not count as completed reviews.")
        root = DATA / "annotations/passage_label_runs" / progress["corpus_sha256"][:16]
        labels = []
        for number, expected_hash in progress["output_sha256"].items():
            path = root / "outputs" / f"batch-{int(number):03d}.json"
            if path.exists() and digest(path.read_bytes()) == expected_hash:
                labels.extend(json.loads(path.read_text(encoding="utf-8"))["labels"])
            else:
                st.warning(f"Batch {number} output has changed; validate it again before viewing.")
        published_path = DATA / "annotations/passage_labels.jsonl"
        published_report_path = DATA / "reports/passage_annotation.json"
        if published_path.exists() and published_report_path.exists():
            published_report = json.loads(published_report_path.read_text(encoding="utf-8"))
            if (digest(published_path.read_bytes()) == published_report["labels_sha256"]
                and published_report["corpus_sha256"] == progress["corpus_sha256"]):
                labels = read_jsonl(published_path)
                st.caption(("Archived label version: " if annotation_archive else "Current label version: ") + published_report["version"])
            else:
                st.warning("The merged label hash or corpus version does not match; only previously validated original batches are shown.")
        if labels:
            coverage = active_label_coverage(labels)
            st.caption(f"Current retrieval passage label coverage: {coverage:,} / {len(current_passages):,}; archived machine labels: {len(labels):,}; currently unlabelled passages: {len(current_passages)-coverage:,}.")
            topic_labels = {"production_supply": "Production / Supply", "opec_policy": "OPEC policy", "sanctions_policy": "Sanctions / Policy",
                            "shipping_logistics": "Shipping / Routes", "inventories": "Inventories", "refining_demand": "Refining / Demand",
                            "prices_market": "Prices / Markets", "statistical_methodology": "Statistical definitions / Methods", "other_energy": "Other energy", "other": "Other"}
            topic = st.selectbox("Passage topic filter", ["all"] + sorted({t for r in labels for t in r["topics"]}),
                                 format_func=lambda t: "All" if t == "all" else topic_labels[t])
            attention = st.checkbox("Show only ambiguous passages / Items needing review")
            selected = [r for r in labels if (topic == "all" or topic in r["topics"]) and (not attention or r["needs_review"])]
            keyword = st.text_input("Passage label search (title / entities / archived summary)").strip().casefold()
            if keyword:
                titles = {pid: p["title"] for pid, p in annotation_map.items()}
                selected = [r for r in selected if keyword in (titles[r["passage_id"]] + " " + r["summary_zh"] + " " + " ".join(r["entities"])).casefold()]
            review_rows = read_jsonl(DATA / "annotations/passage_review.jsonl")
            if review_rows:
                outcome_names = {"all": "All", "confirmed": "Confirmed on second pass", "corrected": "Corrected on second pass", "unresolved": "Unresolved"}
                review_filter = st.selectbox("Second-pass machine review result", list(outcome_names), format_func=outcome_names.get)
                if review_filter != "all":
                    selected = [r for r in selected if r.get("second_pass_outcome") == review_filter]
            st.caption(f"The current filter selects {len(selected)} passages; topics are multi-label, so topic counts cannot simply be added to obtain a passage total.")
            if selected:
                by_id = annotation_map
                position = st.selectbox("Source passage", range(len(selected)),
                                       format_func=lambda i: f"{by_id[selected[i]['passage_id']]['title']} · p.{by_id[selected[i]['passage_id']]['page']} · {selected[i]['passage_id']}")
                label = selected[position]
                passage = by_id[label["passage_id"]]
                st.write(label["summary_zh"])
                st.caption("Topics: " + " / ".join(topic_labels[t] for t in label["topics"]) + " · Status: " +
                           {"observed": "Reported as having occurred", "estimated": "Estimated", "forecast": "Forecast", "announced": "Announced", "effective": "Effective",
                            "recovered": "Recovered", "mixed": "Mixed status", "uncertain": "Uncertain", "not_applicable": "Not applicable"}[label["event_status"]])
                st.caption("Entities (original source names): " + " / ".join(label["entities"]))
                st.caption("Oil task relevance: " + {"direct": "Directly relevant", "background": "Background relevance", "out_of_scope": "Out of scope", "uncertain": "Not yet determined"}[label["oil_relevance"]])
                for reason in label["review_reasons"]:
                    st.warning(reason)
                review = next((r for r in review_rows if r["passage_id"] == label["passage_id"]), None)
                if review:
                    st.info("Second-pass machine review · " + outcome_names[review["outcome"]] + ": " + review["note"])
                    if review["exclude"]:
                        st.warning("This review found the passage unsuitable for strict historical / formal evaluation; the source text is not deleted.")
                    with st.expander("Second-pass review support (verbatim source or context excerpts)"):
                        for context_id, quote in review["support"]:
                            st.caption("Support location: " + context_id + "; PDF text does not establish that table layout has been visually verified.")
                            st.write(quote)
                st.write(passage["text"])
                st.markdown(f"[View original source]({passage['url']}) · Publication availability date {passage['available_by']}")
                with st.expander("Quantities, units, event dates, and original labels"):
                    for quantity in label["quantities"]:
                        st.write(f"Quantity: {quantity['value_text']} {quantity['unit_text']} · Reference period / Definition: {quantity['reference_period']}")
                        st.caption("Source support: " + quantity["quote"])
                    for event in label["events"]:
                        st.write(f"Event: {event['description']} · Source date: {event['date_text'] or 'Not specified'}")
                        st.caption("Source support: " + event["quote"])
                    if not label["quantities"] and not label["events"]:
                        st.caption("No key quantities or dated events were extracted separately; this does not mean the passage is unlabelled or that its text contains no numbers.")
                    st.json(label)
            else:
                st.info("No labelled passages match this filter.")
        else:
            st.info("Batches are prepared, but no model labels have passed validation.")

with status_tab:
    if staged_root:
        st.caption("Explicit staging mode; corpus release and default-library replacement have not occurred.")
    codex_report = DATA / "reports/codex_annotation.json"
    if codex_report.exists():
        st.subheader("Codex machine annotation progress")
        st.json(json.loads(codex_report.read_text(encoding="utf-8")))
    report = DATA / "reports/corpus.json"
    if report.exists():
        st.json(json.loads(report.read_text(encoding="utf-8")))
    if not staged_root:
        st.json(agreement())
    else:
        st.caption("No human-annotation database is connected to this staged query runtime.")
    st.warning("The student reports instructor approval for solo completion and machine annotation. Machine outputs are not human ground truth, and engineering checks alone are not formal evaluation results.")
