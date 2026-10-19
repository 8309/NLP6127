import argparse
import json
import sys

from .store import DATA


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Oil Evidence reproducible pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["discover", "index", "status", "agreement", "audit"]:
        sub.add_parser(command)
    p = sub.add_parser('build', help='Stage a uniform full-source version; never overwrite the live corpus')
    p.add_argument('--run', default='semantic-page-v1')
    for command, default in [("collect", 6), ("pilot", 50), ("generate", 50)]:
        child = sub.add_parser(command)
        child.add_argument("--limit", type=int, default=default)
        if command == "collect":
            child.add_argument("--source", choices=["EIA", "EIA-TIE", "EIA-TWIP", "OFAC", "OPEC"])
    p = sub.add_parser("query")
    p.add_argument("question")
    p.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p.add_argument("--as-of")
    p.add_argument("--region")
    p.add_argument("--no-temporal", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--generation", choices=["extractive", "llm"], default="extractive")
    p = sub.add_parser("export-reviews")
    p.add_argument("--dataset", default="pilot", choices=["pilot", "candidates"])
    p = sub.add_parser("pool")
    p.add_argument("--dataset", default="pilot", choices=["pilot", "candidates"])
    p = sub.add_parser("freeze")
    p.add_argument("path")
    p = sub.add_parser("evaluate")
    p.add_argument("path")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--generation", choices=["extractive", "llm"], default="extractive")
    args = parser.parse_args()
    try:
        if args.command in {"discover", "collect", "build"}:
            from .corpus import build_corpus, collect, discover
            if args.command == 'build':
                # The script is the sole writer workflow; normal builds cannot
                # silently mix old chunks or replace an active index.
                import importlib.util
                from .store import ROOT
                spec = importlib.util.spec_from_file_location('oil_unified_pipeline', ROOT / 'scripts/unify_corpus.py')
                pipeline = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(pipeline)
                pipeline.select_run(args.run)
                with pipeline.lock():
                    result = pipeline.stage()
            else:
                result = discover() if args.command == "discover" else collect(args.limit, args.source)
        elif args.command == "audit":
            from .qa import audit_corpus
            result = audit_corpus()
        elif args.command == "index":
            from .retrieval import build_index
            result = build_index()
        elif args.command == "query":
            from .query import query
            result = query(args.question, mode=args.mode, as_of=args.as_of, region=args.region,
                           temporal=not args.no_temporal, verify=args.verify, generation=args.generation)
        elif args.command in {"pilot", "generate", "agreement", "export-reviews", "freeze", "pool"}:
            from .annotations import agreement, export_reviews, freeze, generate_candidates, prepare_pilot, refresh_pools
            if args.command == "pilot":
                result = prepare_pilot(args.limit)
            elif args.command == "generate":
                result = generate_candidates(args.limit)
            elif args.command == "agreement":
                result = agreement()
            elif args.command == "export-reviews":
                result = export_reviews(args.dataset)
            elif args.command == "pool":
                result = refresh_pools(args.dataset)
            else:
                result = freeze(args.path)
        elif args.command == "evaluate":
            from .evaluation import run_experiments
            result = run_experiments(args.path, generation=args.generation, smoke=args.smoke, limit=args.limit)
        else:
            from .annotations import agreement
            from .store import config, read_jsonl
            path = DATA / "reports/corpus.json"
            result = dict(corpus=json.loads(path.read_text()) if path.exists() else None,
                          model_enabled=config()["llm"]["enabled"], candidates=len(read_jsonl(DATA / "annotations/candidates.jsonl")),
                          pilot=len(read_jsonl(DATA / "annotations/pilot.jsonl")), human_agreement=agreement())
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
