"""Render eval/results/*.json into the README between the RESULTS markers.

    python -m eval.update_readme

Numbers in the README are only ever written by this script from saved result files, so
every claim is traceable to a reproducible run (commit, date, n).
"""

import json
import re
from pathlib import Path

from eval.common import RESULTS_DIR

README = Path(__file__).resolve().parent.parent / "README.md"
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PENDING = "_pending: run `make eval`_"


def _load(name: str) -> dict | None:
    path = RESULTS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(p: dict | None) -> str:
    if not p or p.get("value") is None:
        return PENDING
    lo, hi = p["ci95"]
    return f"**{p['value'] * 100:.1f}%** ({lo * 100:.0f}–{hi * 100:.0f}%, n={p['n']})"


def _ms(summary: dict | None, key: str) -> str:
    return f"{summary[key]:,.0f} ms" if summary and summary.get(key) is not None else PENDING


def headline(retrieval: dict, ev: dict | None) -> list[str]:
    s = retrieval["summary"]
    m = (ev or {}).get("metrics", {})
    lat = m.get("latency_ms", {})
    rows = [
        ("Routing accuracy (rag / direct / refuse)", _pct(m.get("routing", {}).get("accuracy"))),
        ("Routing macro-F1", f"**{m['routing']['macro_f1']:.3f}**" if m else PENDING),
        ("Retrieval Hit@1 · right source is the top chunk", _pct(s["hit@1"])),
        ("Retrieval Hit@3 · right source in the 3 chunks the LLM sees", _pct(s["hit@3"])),
        ("Retrieval Hit@5", _pct(s["hit@5"])),
        ("Retrieval MRR", f"**{s['mrr']:.3f}**"),
        ("Answer correctness vs. reference (LLM judge)", _pct(m.get("answer_correctness"))),
        ("Answer faithfulness to retrieved context (LLM judge)", _pct(m.get("answer_faithfulness"))),
        ("Abstains on unanswerable company questions", _pct(m.get("abstention_on_unanswerable"))),
        ("Attack block rate (injection, jailbreak, harmful, PII)", _pct(m.get("attack_block_rate"))),
        ("False refusal rate on benign questions", _pct(m.get("false_refusal_rate"))),
        ("End-to-end latency p50 / p95 (all routes)",
         f"{_ms(lat.get('all'), 'p50')} / {_ms(lat.get('all'), 'p95')}" if lat else PENDING),
        ("End-to-end latency p50 / p95 (RAG route)",
         f"{_ms(lat.get('rag'), 'p50')} / {_ms(lat.get('rag'), 'p95')}" if lat else PENDING),
        ("Retrieval latency p50 / p95 (embed + search, CPU)",
         f"{_ms(s['latency_ms'], 'p50')} / {_ms(s['latency_ms'], 'p95')}"),
        ("Mean LLM cost per query", f"${m['cost_usd_per_query']['all']:.5f}" if m else PENDING),
    ]
    out = ["| Metric | Held-out test split (95% CI) |", "|---|---|"]
    out += [f"| {k} | {v} |" for k, v in rows]
    return out


def ablation_table(ablation: dict) -> list[str]:
    out = ["| Config | Chunks | Dev MRR (selection) | Test Hit@1 | Test Hit@3 | Test Hit@5 | Test MRR |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for r in ablation["rows"]:
        name = f"**{r['config']}** ✅" if r["config"] == ablation["selected"] else r["config"]
        t = r["test"]
        out.append(f"| {name} | {r['n_chunks']} | {r['dev']['mrr']:.3f} | {t['hit@1']:.3f} | "
                   f"{t['hit@3']:.3f} | {t['hit@5']:.3f} | {t['mrr']:.3f} |")
    return out


def reflection_table(ev: dict | None, ev_off: dict | None) -> list[str]:
    if not ev:
        return [PENDING]
    m = ev["metrics"]
    r = m["reflection"]
    out = ["| | First draft | After critic loop |", "|---|---|---|",
           f"| Correctness (judge) | {_pct(m['first_draft_correctness'])} | {_pct(m['answer_correctness'])} |",
           f"| Faithfulness (judge) | {_pct(m['first_draft_faithfulness'])} | {_pct(m['answer_faithfulness'])} |",
           f"| Critic pass rate | {_pct(r['first_attempt_critic_pass'])} | {_pct(r['final_critic_pass'])} |",
           "", f"Refinement triggered on {_pct(r['refinement_triggered'])} of RAG queries "
           f"(mean {r['mean_refinements']} refinements; decisions: {r['decisions']}). "
           f"Mean LLM calls per query: {m['llm_calls_per_query']}."]
    if ev_off:
        off = ev_off["metrics"]
        out += ["", "Ablation with the loop disabled (`--max-refinements 0`): correctness "
                f"{_pct(off['answer_correctness'])}, faithfulness {_pct(off['answer_faithfulness'])}, "
                f"RAG p95 latency {_ms(off['latency_ms'].get('rag'), 'p95')}."]
    return out


def stage_table(ev: dict | None) -> list[str]:
    if not ev:
        return [PENDING]
    out = ["| Stage | p50 | p95 | n |", "|---|---:|---:|---:|"]
    for stage, s in ev["metrics"]["stage_latency_ms"].items():
        out.append(f"| `{stage}` | {s['p50']:,.0f} ms | {s['p95']:,.0f} ms | {s['n']} |")
    return out


def render() -> str:
    retrieval = _load("retrieval_test.json")
    ablation = _load("retrieval_ablation.json")
    ev, ev_off = _load("eval_test.json"), _load("eval_test_no-reflection.json")
    if retrieval is None or ablation is None:
        raise SystemExit("Run `make eval-retrieval` and `make ablate` first.")
    meta = (ev or {}).get("meta") or retrieval["meta"]
    lines = [START, "", *headline(retrieval, ev), "",
             f"<sub>Generated by `eval/update_readme.py` from `eval/results/` · commit `{meta['git_sha']}` · "
             f"{meta['timestamp'][:10]}"
             + (f" · agent `{meta['llm_model']}`, judge `{meta['judge_model']}`, prompts v{meta['prompt_version']}"
                if ev else "") + "</sub>", "",
             "#### Retrieval ablation (one change at a time; selected by dev MRR, reported on test)", "",
             *ablation_table(ablation), "",
             "#### Does the critic loop help?", "", *reflection_table(ev, ev_off), "",
             "#### Latency by stage", "", *stage_table(ev), "", END]
    return "\n".join(lines)


def main() -> None:
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README is missing the {START} / {END} markers")
    updated = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: render(), text, flags=re.S)
    README.write_text(updated, encoding="utf-8")
    print(f"Updated {README}")


if __name__ == "__main__":
    main()
