"""Portable UI using the packaged corpus, vectors and local query encoder."""
from datetime import date
import json
from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_runtime import load_runtime, search


st.set_page_config(page_title="Oil Evidence Teacher Demo", page_icon="O", layout="wide")
st.title("Oil Evidence Teacher Demo")
st.caption("11,815 English passages. The default evidence preview runs locally without an API key.")


@st.cache_resource
def get_runtime():
    return load_runtime()


try:
    with st.spinner("Checking packaged files and loading the local model..."):
        runtime = get_runtime()
except Exception as exc:
    st.error(f"Cannot load the delivery: {exc}")
    st.stop()

search_tab, labels_tab = st.tabs(["Evidence search", "Test labels"])
with search_tab:
    with st.form("teacher_search"):
        question = st.text_area("English research question",
                                "How do hurricanes affect Gulf of Mexico crude oil production?")
        a, b, c = st.columns(3)
        mode = a.selectbox("Retrieval method", ["hybrid", "bm25", "dense"])
        cutoff = b.date_input("Source cutoff date", value=date(2026, 10, 5))
        region = c.selectbox("Region", ["All", "Middle East", "Europe", "Asia", "Americas", "Africa"])
        generation = st.selectbox("Answer mode", ["Offline source excerpts", "Codex answer using your own local login"])
        verify = st.checkbox("Check generated claims against cited evidence (optional extra model call)", value=False)
        st.caption("Codex mode sends retrieved evidence to your configured Codex service and consumes your own allowance. No account or credentials are included.")
        submitted = st.form_submit_button("Search", type="primary")
    if submitted:
        try:
            if generation == "Offline source excerpts":
                result = search(runtime, question, mode=mode, as_of=cutoff.isoformat(),
                                region=None if region == "All" else region)
            else:
                from oil_evidence.codex_runtime import query_codex
                result = query_codex(question, runtime, mode=mode, as_of=cutoff.isoformat(),
                                     region=None if region == "All" else region, generation="codex", verify=verify)
            st.session_state["teacher_result"] = result
        except Exception as exc:
            st.error(f"Query failed: {exc}")
    result = st.session_state.get("teacher_result")
    if result:
        if result.get("answer_type"):
            st.info("Source excerpt preview only. These excerpts are not a synthesized answer or a guarantee that the question is answerable.")
        for claim in result["claims"]:
            st.write(claim["text"])
            st.caption("Evidence: " + ", ".join(claim["citations"]))
        if not result["evidence"]:
            st.warning("No eligible evidence under these filters.")
        st.subheader("Evidence and source pages")
        for hit in result["evidence"]:
            with st.expander(f"{hit['rank']}. {hit['title']} | {hit.get('available_by', 'Unknown date')}", expanded=True):
                st.write(hit["text"])
                st.caption(f"Page {hit.get('page')} | {hit['passage_id']} | Score {hit['score']:.5f}")
                st.link_button("Open original source", hit["url"])
                with st.expander("Included source page text"):
                    st.text(runtime.pages[hit["source_page_id"]]["text"])
        st.download_button("Download query result", json.dumps(result, indent=2),
                           file_name="oil-evidence-query.json", mime="application/json")
with labels_tab:
    rows = [json.loads(line) for line in (Path(__file__).parent / "data/test_labels.jsonl").read_bytes().splitlines() if line.strip()]
    st.caption("1,000 human-annotated TEST records. Origin confirmed by the project owner on 19 October 2026; see data/LABEL_ORIGIN.json.")
    position = st.number_input("Record number", min_value=1, max_value=len(rows), value=1)
    record = rows[int(position) - 1]
    st.subheader(record["question"])
    st.caption(f"{record['record_id']} | Cutoff: {record.get('as_of')}")
    st.json(record["label"], expanded=True)
