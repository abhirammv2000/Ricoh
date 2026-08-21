"""eval/label_for_kappa.py - Measure whether the LLM judge agrees with a human.

The gap this closes
`eval/judge_variance.py` established the judge is *precise*, it returns the
same score for the same input.  Precision is not accuracy.  A judge can be
perfectly consistent and consistently wrong, and every quality number in this
project inherits that error without ever exposing it.

The only way to find out is to grade some answers yourself and compare.

Why Cohen's κ and not raw agreement
Raw agreement is inflated by the base rate.  When ~90% of answers are good, a
judge that says "good" every single time scores ~90% agreement while carrying
no information at all.  Cohen's κ corrects for agreement expected by chance:

    κ = (p_observed - p_chance) / (1 - p_chance)

    κ ≤ 0     no better than chance
    0.2-0.4   fair
    0.4-0.6   moderate
    0.6-0.8   substantial
    > 0.8     near-perfect

This matters here specifically: published evaluations of LLM judges find raw
agreement overstates chance-corrected agreement by tens of points, so reporting
raw agreement would be the flattering-but-meaningless option.

Scores are binarised at a threshold because κ needs categories. The threshold
is recorded in the output so the number is reproducible rather than tuned.

Workflow
    python -m eval.label_for_kappa --sample 30    # writes a worksheet
    # ... fill in the "human_*" fields by hand ...
    python -m eval.label_for_kappa --score        # computes κ

Sample honestly: the worksheet deliberately HIDES the judge's scores while you
label, because seeing them first would anchor you and inflate agreement.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

WORKSHEET_PATH: Path = PROJECT_ROOT / "eval" / "human_labels.json"

# A score at or above this counts as "acceptable" for the binary comparison.
# 0.8 is the point below which an answer has a material defect (a claim not in
# the evidence, or a key fact missed) rather than a stylistic shortfall.
BINARY_THRESHOLD = 0.8


def make_worksheet(metrics_path: Path, n: int, seed: int) -> int:
    report = json.loads(io.open(metrics_path, encoding="utf-8").read())
    rows = [r for r in report["per_question"] if r.get("groundedness") is not None]
    if not rows:
        raise SystemExit(f"No judged rows in {metrics_path}")

    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))

    items = []
    for r in sample:
        items.append(
            {
                "id": r["id"],
                "question": r["question"],
                "answer": r["answer"],
                "evidence_docs": r.get("evidence_docs", []),
                "expected_behavior": r.get("expected_behavior"),
                # --- FILL THESE IN: 1 = acceptable, 0 = not acceptable ---
                "human_grounded": None,
                "human_correct": None,
                "human_note": "",
                # Judge scores are withheld until scoring, to avoid anchoring.
                "_judge_scores_hidden": True,
            }
        )

    payload = {
        "_instructions": (
            "For each item set human_grounded and human_correct to 1 or 0. "
            "grounded = every claim in the answer is supported by the cited "
            "documents. correct = the answer actually answers the question "
            "(or correctly refuses, when expected_behavior is 'refuse'). "
            "The judge's own scores are deliberately not shown here, seeing "
            "them first would anchor your labels and inflate agreement. "
            "Then run: python -m eval.label_for_kappa --score"
        ),
        "source_metrics": str(metrics_path.name),
        "binary_threshold": BINARY_THRESHOLD,
        "seed": seed,
        "items": items,
    }
    WORKSHEET_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(items)} items to {WORKSHEET_PATH}")
    print("Fill in human_grounded / human_correct (1 or 0), then:")
    print("  python -m eval.label_for_kappa --score")
    return 0


def _kappa(human: list[int], judge: list[int]) -> dict[str, float]:
    n = len(human)
    observed = sum(1 for h, j in zip(human, judge) if h == j) / n

    # Chance agreement from each rater's marginal rates.
    ph1 = sum(human) / n
    pj1 = sum(judge) / n
    chance = ph1 * pj1 + (1 - ph1) * (1 - pj1)

    kappa = (observed - chance) / (1 - chance) if chance < 1 else float("nan")

    # How much one flipped item would move kappa.
    #
    # When almost every item falls in one category, chance agreement is huge
    # and the denominator (1 - chance) is tiny, so kappa swings wildly on a
    # single disagreement. A kappa of 1.0 sitting on 94% chance agreement is
    # not the same evidence as a kappa of 1.0 sitting on 50%, and reporting
    # them identically would be the flattering read.
    if n > 1 and chance < 1:
        one_flip = (observed - 1 / n - chance) / (1 - chance)
        fragility = round(abs(kappa - one_flip), 3)
    else:
        fragility = float("nan")

    return {
        "n": n,
        "raw_agreement": round(observed, 3),
        "chance_agreement": round(chance, 3),
        "cohens_kappa": round(kappa, 3),
        "kappa_drop_if_one_item_flipped": fragility,
        "human_positive_rate": round(ph1, 3),
        "judge_positive_rate": round(pj1, 3),
    }


def _interpret(k: float) -> str:
    if k != k:  # NaN
        return "undefined (one rater used a single category throughout)"
    if k <= 0:
        return "no better than chance, the judge is not usable as evidence"
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "near-perfect"


def score() -> int:
    if not WORKSHEET_PATH.exists():
        raise SystemExit(f"No worksheet at {WORKSHEET_PATH}. Run --sample first.")
    sheet = json.loads(io.open(WORKSHEET_PATH, encoding="utf-8").read())
    metrics_path = PROJECT_ROOT / "eval" / sheet["source_metrics"]
    report = json.loads(io.open(metrics_path, encoding="utf-8").read())
    judged = {r["id"]: r for r in report["per_question"]}
    threshold = sheet.get("binary_threshold", BINARY_THRESHOLD)

    labelled = [
        it for it in sheet["items"]
        if it.get("human_grounded") is not None and it.get("human_correct") is not None
    ]
    if len(labelled) < 10:
        raise SystemExit(
            f"Only {len(labelled)} items labelled. Label at least 10, below that, "
            "kappa is too unstable to mean anything."
        )

    results: dict[str, Any] = {"labelled": len(labelled), "threshold": threshold}
    print(f"Labelled items: {len(labelled)} / {len(sheet['items'])}")
    print(f"Binary threshold: score >= {threshold} counts as acceptable\n")

    for metric, human_key in (("groundedness", "human_grounded"), ("correctness", "human_correct")):
        human, judge, disagreements = [], [], []
        for it in labelled:
            row = judged.get(it["id"])
            if row is None or row.get(metric) is None:
                continue
            h = int(it[human_key])
            j = 1 if row[metric] >= threshold else 0
            human.append(h)
            judge.append(j)
            if h != j:
                disagreements.append((it["id"], h, j, row[metric], it["question"][:60]))

        if not human:
            continue
        stats = _kappa(human, judge)
        results[metric] = stats
        print(f"{metric}")
        print(f"  raw agreement  : {stats['raw_agreement']:.1%}")
        print(f"  chance agreement: {stats['chance_agreement']:.1%}")
        print(f"  Cohen's kappa  : {stats['cohens_kappa']}  ({_interpret(stats['cohens_kappa'])})")
        print(f"  positives - human {stats['human_positive_rate']:.0%} / judge {stats['judge_positive_rate']:.0%}")
        if stats["raw_agreement"] - stats["chance_agreement"] < 0.1:
            print("  raw agreement is close to chance, the high raw number is a")
            print("    base-rate artifact, not evidence the judge is accurate.")
        if disagreements:
            print(f"  disagreements ({len(disagreements)}):")
            for qid, h, j, raw, q in disagreements[:8]:
                print(f"    Q{qid}: human={h} judge={j} (raw {raw:.2f}), {q}")
        print()

    out = PROJECT_ROOT / "eval" / "judge_calibration.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Written to {out}")
    return 0


# A different model, comparable capability, and NOT the agent under test.
# Using the agent's own model would reintroduce the self-preference bias the
# independent judge exists to avoid.
CROSS_JUDGE_MODEL = "claude-opus-4-8"

_STOP = set(
    "the a an of to and or is are was were be been for in on at by with from as "
    "that this it its you your they their we our i my if then than so not no "
    "do does did can could should would will shall may might must have has had "
    "when where which who whom what how why all any each both more most other "
    "some such only own same too very s t just don now".split()
)


def _stem(w: str) -> str:
    """Crude suffix stripping.

    Not linguistics, just enough that "use"/"used"/"using" and
    "name"/"names" stop counting as different words. Without it the matcher
    reported false defects on answers that plainly stated the fact.
    """
    for suf in ("ing", "ed", "es", "s"):
        if len(w) > 4 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _content_words(text: str) -> set[str]:
    import re as _re

    return {
        _stem(w) for w in _re.findall(r"[a-z0-9_]+", text.lower())
        if w not in _STOP and len(w) > 1
    }


# A key fact counts as present when this share of its content words appear in
# the answer. Requiring ALL of them (1.0) was the original setting and it
# produced 17 false defects out of 100, every one an answer that stated the
# fact in different words. Natural paraphrase reorders and re-inflects, so an
# exact-set test measures wording, not meaning.
FACT_COVERAGE_THRESHOLD = 0.7


def _fact_present(fact: str, answer_words: set[str]) -> bool:
    fw = _content_words(fact)
    if not fw:
        return False
    return len(fw & answer_words) / len(fw) >= FACT_COVERAGE_THRESHOLD


def spot_check(metrics_path: Path) -> int:
    """Deterministic key-fact coverage, with no LLM in the loop.

    Why this is worth having alongside the judge
    Every LLM-based check in this project shares a failure mode: the judge and
    the agent are both Claude models, so a blind spot they share is invisible
    to cross-model agreement.  This check has no model in it at all, it asks
    only whether the answer literally contains the content words of each
    curated key fact.

    That makes it narrow but unbiased.  It cannot assess faithfulness or
    reasoning; it can catch a judge awarding full correctness to an answer
    that never states the fact it was supposed to state.

    Disagreements are the output that matters.  Agreement here is weak
    evidence (both could be right for different reasons); a judge score of
    1.00 on an answer containing none of the key facts is a concrete defect
    in either the judge, the answer, or the key facts themselves.
    """
    report = json.loads(io.open(metrics_path, encoding="utf-8").read())
    gt_path = PROJECT_ROOT / "eval" / "generated_questions.json"
    truth = {}
    if gt_path.exists():
        for q in json.loads(io.open(gt_path, encoding="utf-8").read())["questions"]:
            truth[q["id"]] = q.get("key_facts", [])

    rows, flagged = [], []
    for r in report["per_question"]:
        facts = truth.get(r["id"], [])
        if not facts:
            continue
        answer_words = _content_words(r.get("answer", ""))
        present = sum(1 for f in facts if _fact_present(f, answer_words))
        coverage = present / len(facts)
        rows.append((r["id"], coverage, r.get("correctness")))
        if coverage == 0.0 and (r.get("correctness") or 0) >= 0.8:
            flagged.append((r["id"], r["question"], r.get("correctness")))

    if not rows:
        raise SystemExit("No questions with key_facts found.")

    mean_cov = sum(c for _, c, _ in rows) / len(rows)
    full = sum(1 for _, c, _ in rows if c == 1.0)
    print(f"Key-fact coverage (deterministic, no LLM) over {len(rows)} questions")
    print(f"  mean coverage        : {mean_cov:.1%}")
    print(f"  all facts present    : {full}/{len(rows)}")
    print(f"  judge mean correctness: {sum((c or 0) for _, _, c in rows)/len(rows):.3f}")
    print()
    if flagged:
        print(f"{len(flagged)} answer(s) scored >= 0.8 by the judge with ZERO key facts present:")
        for qid, q, c in flagged[:10]:
            print(f"  Q{qid} (correctness {c}): {q[:76]}")
        print("\n  Each is a real defect in one of three places: the judge is wrong,")
        print("  the answer is wrong, or the key_facts are badly written. Worth reading.")
    else:
        print("No answer was credited by the judge while containing none of its key facts.")
    print("\nNote: exact-wording matching. A correctly paraphrased fact counts as")
    print("absent here, so coverage UNDERSTATES quality. Only the disagreements")
    print("are informative, not the absolute number.")
    return 0


def cross_judge(metrics_path: Path, n: int, seed: int) -> int:
    """Agreement between two different judge models.

    What this does and does not establish
    It catches a judge that is idiosyncratic, unstable, or reacting to
    artefacts of one prompt formulation.

    It does NOT establish accuracy.  Both judges are Claude models, so any
    bias they share, over-crediting confident prose, under-penalising a
    subtly unsupported claim, produces high agreement between two judges
    that are both wrong.  A strong kappa here is therefore *weaker* evidence
    than it looks, and is reported as a consistency check rather than as
    validation.  Only human labels close that gap.
    """
    from src.eval_harness import _format_evidence_block, _judge
    from src.instrumentation import record_run
    from src.retriever import get_retriever

    report = json.loads(io.open(metrics_path, encoding="utf-8").read())
    cfg = report.get("config", {})
    if cfg.get("use_planner"):
        print("This run used the planner, so retrieval is not reproducible from")
        print("  the question alone and the reconstructed evidence may differ.")

    rows = [r for r in report["per_question"] if r.get("correctness") is not None]
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    retriever = get_retriever()
    threshold = BINARY_THRESHOLD

    import src.config as _config

    original = _config.JUDGE_MODEL
    primary, secondary, disagreements = [], [], []

    with record_run(persist=False) as rec:
        for r in sample:
            ev = retriever.retrieve(
                query=r["question"],
                top_k=_config.RETRIEVAL_TOP_K,
                final_k=_config.RETRIEVAL_FINAL_K,
            )
            block = _format_evidence_block(ev)
            # Swap the judge model for this call only.
            import src.eval_harness as _eh

            _eh.JUDGE_MODEL = CROSS_JUDGE_MODEL
            try:
                v = _judge(r["question"], r["answer"], block, [], r.get("expected_behavior", "answer"))
            finally:
                _eh.JUDGE_MODEL = original

            p = 1 if r["correctness"] >= threshold else 0
            s = 1 if v["correctness"] >= threshold else 0
            primary.append(p)
            secondary.append(s)
            if p != s:
                disagreements.append((r["id"], r["correctness"], v["correctness"], r["question"][:60]))

    stats = _kappa(primary, secondary)
    print(f"Cross-judge agreement: {original} vs {CROSS_JUDGE_MODEL}")
    print(f"  n                : {stats['n']}")
    print(f"  raw agreement    : {stats['raw_agreement']:.1%}")
    print(f"  chance agreement : {stats['chance_agreement']:.1%}")
    print(f"  Cohen's kappa    : {stats['cohens_kappa']}  ({_interpret(stats['cohens_kappa'])})")
    print(f"  fragility        : one flipped item would move kappa by "
          f"{stats['kappa_drop_if_one_item_flipped']}")
    print(f"  cost             : ${rec.total_cost_usd:.4f}")
    if stats["chance_agreement"] > 0.85:
        print()
        print("  Chance agreement is very high because both judges rate almost")
        print("    everything acceptable. Kappa has little room to move, so this")
        print("    result is far weaker evidence than the headline number suggests.")
    if disagreements:
        print(f"\n  {len(disagreements)} disagreement(s):")
        for qid, a, b, q in disagreements[:8]:
            print(f"    Q{qid}: {original}={a:.2f} {CROSS_JUDGE_MODEL}={b:.2f}, {q}")

    out = PROJECT_ROOT / "eval" / "judge_cross_check.json"
    out.write_text(
        json.dumps(
            {
                "primary_judge": original,
                "secondary_judge": CROSS_JUDGE_MODEL,
                "threshold": threshold,
                "cost_usd": rec.total_cost_usd,
                "caveat": (
                    "Consistency check only. Both judges are Claude models, so a "
                    "shared bias yields high agreement while both are wrong. This "
                    "does not substitute for human labels."
                ),
                **stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWritten to {out}")
    print("\nRead this as a CONSISTENCY check, not validation: two Claude models")
    print("sharing a blind spot agree with each other and are both wrong.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Judge-vs-human agreement (Cohen's kappa)")
    ap.add_argument("--cross-judge", action="store_true", help="agreement between two judge models")
    ap.add_argument("--spot-check", action="store_true", help="deterministic key-fact coverage, no LLM")
    ap.add_argument("--n", type=int, default=30, help="sample size for --cross-judge")
    ap.add_argument("--sample", type=int, help="create a worksheet with N items")
    ap.add_argument(
        "--metrics", type=Path,
        default=PROJECT_ROOT / "eval" / "metrics_n100.json",
        help="metrics file to sample answers from",
    )
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--score", action="store_true", help="compute kappa from a filled worksheet")
    args = ap.parse_args()
    if args.spot_check:
        raise SystemExit(spot_check(args.metrics))
    if args.cross_judge:
        raise SystemExit(cross_judge(args.metrics, args.n, args.seed))
    if args.score:
        raise SystemExit(score())
    if args.sample:
        raise SystemExit(make_worksheet(args.metrics, args.sample, args.seed))
    ap.error("pass --sample N, --score, --cross-judge, or --spot-check")
