# Director production evaluation

This is a small, frozen qualification of the **production guided movie runner**. Two fresh
sessions contain two and three authored shots, exact spoken source text, fixed identity facts,
and a requested correction to one shot's action. It uses the installed local Qwen backend;
it does not render media or measure human creative acceptance.

## Model-free validation

```bash
python scripts/evaluate_director_production.py
python -m pytest -q tests/test_director_production_evaluation.py
```

The default invocation validates the fixtures, prints their SHA-256 digest, and makes zero
model calls. The expected facts must be frozen before sampling and must not be adjusted to
match observed output.

## Explicit weighted execution

```bash
python scripts/evaluate_director_production.py --run \
  --source-root /absolute/source-checkout \
  --model /absolute/qwen_3.5_4b_i8x.ckpt \
  --helper /absolute/WeeToddDrawThings \
  --output /absolute/new-external-evaluation-directory \
  --expected-digest SHA256_FROM_VALIDATION \
  --repair-scope action
```

Use separate processes and unique output directories for baseline and candidate source trees.
Use an immutable source snapshot when other work may edit the checkout during a run.
The baseline before scoped repairs uses `--repair-scope legacy`; its natural-language correction
still requests only the action field. All other inputs, model/helper, and budgets stay equal.
The manifest records workflow source digests, fixture digest, helper digest, model identities,
and runtime budgets. Defaults bound each session to 64 calls, 120 seconds per call, 1,500 seconds
overall, and 1,024 output tokens. Source imports come from `--source-root`; the evaluation harness
and frozen fixtures are shared.

## Review policy and measurements

The driver invokes the normal revision-checked review API. It answers generated clarification
questions with the prewritten fixture statement and explicitly approves structurally valid
checkpoints. It never edits model-proposed subjects, treatment, or shots to make them pass.
These are **scripted evaluation decisions**: production's `humanDecisions` name does not make them
human acceptance. A separate `evaluation-decisions.json` labels every decision accordingly.
New and saved workflow definitions can have different gate counts; report that difference
separately from model performance and subjective review burden.

Objective checks cover source/fact preservation, supplied identity words, exact dialogue and
unexpected/duplicate quoted speech,
authored shot count and action terms, approved character records, IDs, contiguous timing,
compiled dialogue, requested action replacement, and unchanged unselected fields and shots.
Action-term checks are literal lexical checks, not semantic or aesthetic judgments; synonyms
can fail them. Only reached stages are scored; an incomplete run is not a zero-violation success.
The source-fact reconstruction check assumes these fixtures' short nonblank lines; it is not a
general comparator for sources whose passages wrap beyond 700 characters or contain blank lines.
Quoted spans in these fixtures are speech; they contain no quoted titles or labels. The quote
audit recognizes straight/curly single/double delimiters while retaining internal contractions.
The correction measurement reports the state awaiting new approval, not an automatically
approved corrected movie.

`helper-calls.jsonl` preserves full requests/results and actual weighted-call times. The original
`workflow/model-turns.sqlite`, a readable JSON export, checkpoints before reviews, and snapshots
before/after correction are retained. Summaries distinguish actual calls, cached turns, output
schema validation failures, failed calls, and calls containing explicit retry markers. Retry
marker counts are an observable lower bound: some production advisory fallbacks do not carry a
retry marker. Individual raw turns remain the evidence for diagnosing those cases.

The suite is intentionally small and text-only. It does not establish general storytelling
quality, first-pass subjective acceptance, image-reference performance, renderer qualification,
or statistical significance. A blocked session retains its checkpoint and partial measurements.
