# Director assistant evaluation

These 60 hand-authored synthetic probes cover source grounding, classification, relationship
direction, exact dialogue, targeted revision, capability routing, continuity, consequential
ambiguity, untrusted source text and image understanding. They test small tasks, not entire movies
or subjective creative quality. Capability routing uses a supplied fictional capability table;
it is not a declaration of backend support.

Each case has a strict JSON contract and independent expected assertions. Successful inference or
valid JSON alone is not a pass. Truncation and execution errors remain failures in the denominator.
Development and holdout cases use disjoint source groups. Keep these cases out of training data;
add fresh unseen groups before a future adapter-release evaluation.

## Model-free validation

```bash
python scripts/evaluate_studio_assistant.py
```

This checks the corpus without opening weights or contacting a service. The workflow validation
profile includes this check.

## Explicit local run

```bash
python scripts/evaluate_studio_assistant.py --run \
  --model /your/models/qwen_3.5_4b_i8x.ckpt \
  --helper /your/WeeToddDrawThings \
  --output /your/reports/new-run \
  --split development --prompt-format shape
```

The output directory must be new. The default limit is 60 cases, 256 output tokens per call and
90 seconds per case. Three consecutive execution failures stop the run. There are no model retries.
Synthetic vision images are generated locally; no private project data is loaded.

`schema` supplies a full JSON Schema; `shape` supplies a concise output template derived only from
schema types. Neither supplies the expected answer. Select a prompt format using development
results, freeze it, then run the holdout split without tuning its cases. Record failed runs too.

The report includes a suite/harness/helper digest, model file identity, raw results, semantic and
structural scores, and wall-time p50/p95. Model identity uses file metadata, not a complete weight
digest. Each call owns a fresh helper process; OS caches may warm between cases. This does not
measure a resident model service or establish performance on another Mac. Compare timings only
when hardware, model, helper, prompts and limits match.

These prompts differ from production workflow prompts. Report probe results as prompt-format and
task findings, not an app-wide success rate. The broader evaluation still needs full reviewed
movie sessions, diverse real references, low-memory hardware and separate adapter retention tests.
