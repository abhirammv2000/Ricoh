#!/usr/bin/env bash
#
# The four judged evaluation runs that need a funded ANTHROPIC_API_KEY (both the
# synthesizer and the opus judge). All four have been run once and their
# results are written up in README section 7; this script reproduces them.
# Steps are independent; one failing does not stop the others.
#
#   bash eval/run_paid_batch.sh
#
# Actual cost of the first run, 2026 prices:
#   1  cross-provider bakeoff, judged, n=20         $1.80
#   2  multi-hop ablation A/B/C, judged, 20 qs      $4.71
#   3  multi-turn conversation eval, judged         $1.67
#   4  judged holdout ablation A/B, 30 qs           $3.51
#   total                                           ~$11.7
#
# Outputs:
#   eval/provider_bakeoff.md
#   eval/ablation/multihop_questions/  (A/B/C reports + comparison.json)
#   eval/multiturn_report.md
#   eval/ablation/generated_questions_holdout/
set -u
PY=./.venv/Scripts/python.exe
step() { echo; echo "==================== $* ===================="; echo; }

step "0/4 credit check"
$PY - <<'EOF'
import src.config
from src.llm_factory import get_llm
try:
    get_llm(max_tokens=8).invoke("hi")
    print("Anthropic API reachable and funded.")
except Exception as e:
    raise SystemExit(f"Anthropic call failed, stopping: {e}")
EOF
[ $? -ne 0 ] && exit 1

step "1/4 cross-provider bakeoff (judged, n=20)"
$PY -m eval.provider_bakeoff --n 20 || true

step "2/4 multi-hop ablation A/B/C (judged, 20 questions)"
$PY -m eval.ablation --ground-truth eval/multihop_questions.json --configs A B C || true

step "3/4 multi-turn conversation eval (judged)"
$PY -m eval.multiturn_eval || true

step "4/4 judged holdout ablation A/B (30 questions)"
$PY -m eval.ablation --ground-truth eval/generated_questions.json --split holdout --configs A B || true

step "done"
ls -la eval/provider_bakeoff.md eval/multiturn_report.md 2>/dev/null
ls -la eval/ablation/multihop_questions/ 2>/dev/null
