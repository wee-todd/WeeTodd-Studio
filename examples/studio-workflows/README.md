# Authoring WeeTodd Studio workflows and adapters

Workflows belong to the standalone WeeTodd Studio product. They organize creative work around
explicit model/backend choices: Draw Things for supported inference, native MLX for additional
models and tasks, and shared installed weights where compatible. The maintained ComfyUI integration
is a separate interface to the same generation adapters; Studio workflows do not require ComfyUI.
See the [project direction](../../README.md#why-weetodd-studio).

Built-in and user-authored definitions use the same versioned contracts and validator. These
files describe a larger process than an individual render recipe. They do not replace
`weetodd-headless-v2` recipes or portable Draw Things generation requests.

**Current milestone: local execution and Studio controls.** All registered operations
have handlers. Staged prompt editing and movie planning run through the existing local Qwen
helper, with typed outputs, atomic checkpoints, pause/resume and dependency invalidation.
Open **Director** in the toolbar, **Movie → Workflows…**, or **Prompt Assistant → Step-by-step…**. Built-ins and imported
JSON definitions use the same validator and runner. Movie outputs are plans and endpoint
**descriptions**, not generated frames or timeline clips. Qwen task-adapter loading, training
and reviewed dataset export remain future work.

Director keeps unfinished intake and review drafts separate from execution checkpoints. It restores
them on reopening, validates imported jobs before replacing the session, and retains completed
outputs after cancellation or failure. **Set up assistant…** can download the pinned Qwen3.5 4B
checkpoint independently of the Draw Things app; see [setup](../../studio/README.md#local-qwen35-prompt-assistant).

## Try the examples

Use the project's configured Python environment, with project dependencies installed:

```bash
python scripts/validate_studio_workflow.py examples/studio-workflows/staged-prompt-editing.json
python scripts/validate_studio_workflow.py examples/studio-workflows/movie-planning.json --json
```

Both report `valid: true` and `executionStatus: requires_bindings`. Exit status is 0 for a valid
format, 1 for an invalid definition/import, and 2 when `--require-runnable` was requested but
execution is unavailable. The command never generates media, downloads models, opens weights,
contacts an endpoint or runs commands embedded in a document.

Copy either example to start a user workflow. Give it a new namespaced `id`, name and version;
edit its inputs, bindings and supported parameters; then validate it with the same command.
The CLI accepts a definition at any local path and optional `--adapter path/to/manifest.json`
arguments. All manifests are validated before compatibility declarations are compared.

| Example | Steps | Result contract |
| --- | --- | --- |
| [Create a movie](guided-movie-planning.json) | Friendly creative brief → clarify identities → classify and review reusable subjects → treatment and timed shots → H3 prompt drafts | Approved creative choices, subject IDs, shot plan and three-field H3 prompt preview; no media generation |
| [Staged prompt editing](staged-prompt-editing.json) | Describe images → plan discrete edits → apply edits individually → review | Revised prompt and a separate review |
| [Subject inventory](subject-inventory.json) | Identify characters, props and locations in bounded source sections → review | Factual descriptions, selected source passages and optional suggestions; import into Project Subjects for individual approval |
| [Movie planning](movie-planning.json) | Observe references → structured story → approve → clip states → approve → assemble endpoints → structural review | Clip plan, first/last-frame descriptions and review; no generated frames or movie |

## Schemas and versioning

`weetodd.guided-movie-planning` is Studio's default for new movie plans. Version 1.1.0 groups
required review into `creative_brief`, `subjects_coverage` and `clips`: Brief, Subjects and Shots.
Classification, relationships, design, coverage, treatment and prompt compilation still execute
and remain inspectable. Studio's detailed review option restores separate intermediate gates for
new jobs. Saved older definitions retain their original steps and approval requirements.
Identity questions require an answer; creative preferences can delegate to the director.
Only selected answers are passed to extraction, so rejected alternatives do not become facts.
The guided extractor `project.identify_creative_subjects@1` is separate from the legacy extractor.
It asks for compact names and evidence selections; the host validates cited source phrases and
assigns identity. Detailed appearance belongs to the later reviewed design stage. Guided treatment
uses compact action arrays with host-owned cast records. `movie.plan_creative_beats@1` carries
approved object descriptions and creative preferences into each shot while reusing the existing
timing and repair machinery. Locations must match the approved inventory.
Classification preserves app-issued IDs and allows the reviewer to change an unresolved location
to an environment or set. Set/environment relationships use the ordinary ID link contract.

`apply_library_definition` is an explicit guided subject review action. Supply `stepID`, `itemID`,
`expectedRevision`, the current `library` catalog and a `libraryChoice` containing `objectID`,
`packageID`, `version`, `scope`, and `definitionRevision`. The host verifies the candidate, records
an immutable `reusedDefinition` marker and uses that definition during design. A null choice
restores the extracted description. Import rechecks the actual target definition and dependencies.
Neither selection nor agent review approves the subject automatically.

`movie.compile_h3_prompts@1` returns `h3_prompt_preview`: a draft status, warnings and per-clip
`integrated_multimodal_description`, `overall_soundscape`, `non_diegetic_music`, combined `prompt`,
subject IDs and reference asset IDs. It resolves approved descriptions and dependencies into the
text after `movie.check_plan@2` completes. Structural errors block compilation and advisory findings
remain visible in the preview. These are prompt drafts, not executable render recipes; frame conditioning, speaker mapping,
model compatibility and actual generation remain separate controls.

The normative structural contracts are packaged with Python so validation also works away
from a source checkout:

- [Workflow](../../src/wee_todd_mlx/workflows/schemas/workflow-v1.schema.json): package identity, user inputs, logical models, steps, outputs and limits.
- [Step](../../src/wee_todd_mlx/workflows/schemas/step-v1.schema.json): registered operation, typed bindings, parameters, model/adapter references, timeout and retry bounds.
- [Values](../../src/wee_todd_mlx/workflows/schemas/values-v1.schema.json): data exchanged between steps.
- [Adapter](../../src/wee_todd_mlx/workflows/schemas/adapter-v1.schema.json): model/weight/training metadata, separate from workflows and model files.
- [Operation registry](../../src/wee_todd_mlx/workflows/operations.json): supported operation versions, ports and parameter schemas.

Schema dialect is JSON Schema 2020-12. Schema identifiers use `urn:weetodd:schema:…:1` and are
resolved from packaged resources only. No external schema fetching is performed.

`format` is exactly `weetodd-workflow-v1` or `weetodd-adapter-v1`. Package `version` uses numeric
`major.minor.patch`. IDs are stable namespaced strings, such as `yourname.movie-planner`.
An operation is pinned to a contract version, for example `text.apply_edits@1`. An incompatible
operation or document change requires a new contract version. Unknown operations, versions and
properties are rejected rather than silently dropped. Package versions are metadata in this
milestone; automatic updates, migrations, distribution and dependency locking are future work.

## Inputs, bindings and dataflow

Workflow inputs declare a `type`, human-readable `label`, optional `default`, and optional
numeric `minimum`/`maximum`. Types available to users are `text`, `integer`, `number`, `boolean`
and `image_list`. Defaults are validated. Images use logical asset IDs such as
`asset:character-front`; the job’s `assets` mapping resolves them to existing files. Definitions
contain neither machine-specific media paths nor credentials. Image lists contain at most eight
references for the initial Qwen vision path.

A step connects an input using one of two unambiguous forms:

```json
{"input": "instructions"}
```

```json
{"step": "plan", "output": "edits"}
```

These are references, not expressions or string substitutions. Dependencies are inferred from
bindings. IDs must be unique; every input/output must exist; types must agree; cycles are rejected.
An integer may feed a number port, but a number may not feed an integer port. Steps can be written
in any order; the report supplies a dependency-correct order. Workflow outputs use the same
binding syntax. Files cannot introduce custom executable operation implementations.

The validator enforces declared retry bounds (1–3 attempts), timeouts (1–3,600 seconds), finite
iteration limits, and a sufficient `maxStepExecutions` budget. `maxParallelSteps` is fixed at 1 for
this first contract. `maxArtifacts` and `maxWorkingBytes` bound runner-owned checkpoint storage; they do not
change existing renderers. The runner requires at least three artifact slots for its state,
atomic temporary and lock. It caps the state at 2 MiB and reserves twice its serialized size
for atomic replacement. Model calls require five slots, including the full-turn database and
its transaction journal, as described under execution history below. Models and media are
never copied into run directories.
`maxWorkingBytes` refers to temporary artifact storage, not model weights or resident GPU memory.

## Operation contracts

All operations below are implemented by the shared local runner:

| Operation | Required inputs | Outputs | Parameters / responsibility |
| --- | --- | --- | --- |
| `vision.describe@1` | `images: image_list` | `observations: observation_list` | Describe each image separately; preserve its asset ID and record uncertain details. Empty images produce empty observations. Requires vision. |
| `text.plan_edits@1` | `source: text`, `instructions: text` | `edits: edit_list` | `maxEdits` 1–32. Split instructions into explicit edits and preservation requirements; report an over-budget plan instead of truncating it. Requires text. |
| `text.apply_edits@1` | `source: text`, `edits: edit_list`, `observations: observation_list` | `draft: text` | `maxEdits` 1–32. Apply one edit per model call, carrying the revised draft into the next call. Requested changes override conflicting source/image details. Requires text. |
| `text.check_edits@1` | `source: text`, `draft: text`, `edits: edit_list` | `review: review` | Check requested edits, preservation and format. A review is separate from the proposed text and does not automatically certify model correctness. Requires text. |
| `movie.plan_story@1` | `brief: text`, `duration_seconds: number` | `story: text` | Plan the story progression for the requested length. Requires text. |
| `movie.allocate_clips@1` | `story: text`, `duration_seconds: number`, `target_clip_seconds: number`, `frame_rate: integer` | `clips: clip_plan` | `maxClips` 1–1,000. Deterministic timeline allocation and chronological action scoping; report impossible requests, invalid story time coverage and frame-rounding. No model. |
| `movie.plan_endpoints@1` | `clips: clip_plan`, `images: image_list` | `endpoints: endpoint_plan` | `maxClips` 1–1,000. Describe first/last frames one clip at a time, with explicit endpoint reuse. Keep character references distinct from requested motion. Requires vision. |
| `movie.check_plan@1` | `clips: clip_plan`, `endpoints: endpoint_plan`, `duration_seconds: number`, `frame_rate: integer` | `review: review` | Check clip coverage, duration, ownership and continuity links; warn about repeated actions/endpoints and possible premature endings. No model. |

The legacy Movie planning definition is version **2.0.0**; the document format stays v1. Existing saved
v1 definitions keep their operation versions. New operations are:

| Operation | Inputs → outputs | Responsibility |
| --- | --- | --- |
| `movie.plan_story@2` | brief, duration, target clip length, FPS, observations → `story_outline` | Fixed character IDs/descriptions and exactly one action per clip; at most four actions per writing batch. No model-generated timestamps. |
| `movie.plan_beats@1` | outline, duration, target clip length, FPS → `structured_clip_plan` | Exact frame allocation plus one bounded JSON task per clip. At most two attempts per clip, inside the workflow call/time budget. |
| `movie.plan_endpoints@2` | `structured_clip_plan` → `endpoint_plan` | Compose descriptions from cast, location and visible states. Continuous first frames reuse the previous last frame. No model call. |
| `movie.check_plan@2` | structured clips, endpoints, duration, FPS → `planning_review` | Exact timing, character IDs, continuity and pacing warnings. `structure: valid/invalid` and `storyReview: human_required` are separate from execution completion and human approval. |

Detailed scripts may use consecutive `[Shot 1]`, `[Shot 2]`, … headings. When their count
matches the planned clip count, each shot is summarized separately in source order instead of
being rewritten through a shorter outline. Leading timestamps such as `At 00:05.000` must agree
with project-frame starts (within half a frame); count/timing conflicts report an actionable error
before model calls. `overall_soundscape:`, `non_diegetic_music:` and `negative_prompt:` sections
remain in the original saved input but are not mistaken for another shot's visible action.
These plans condense action and character descriptions; the complete source script stays in
`run.json` and the job. Review the summaries before approval. They are not a lossless renderer
recipe for every lighting, dialogue or music direction.

Model JSON parsing can restore **one missing closing brace or bracket at the outermost level**
when strings and inner containers are already complete. It then applies the unchanged duplicate-key,
finite-number, schema and semantic checks. It never invents strings, fields or separators;
unclosed strings, incomplete nested containers, malformed syntax and flagged truncated responses
still fail. The latest rejected model response is retained in the step record for diagnosis, and
errors distinguish model-output syntax from the user's unchanged input text.

`story_outline` holds `characters: [{id, description}]` and `beats: [string]`, with exactly one distinct action per planned clip (up to 200 in the built-in). Larger stories start from four phases and expand only the current phase in batches of at most four actions. Each action stays intact; an underspecified or duplicate plan fails for correction.
`structured_clip_plan` adds the cast plus each clip's `startState`, `endState`, `location`,
character IDs and `continuity` to the existing frame/action fields. Continuing clips must copy
the previous ending state and location; scene changes use `cut`. The planner checks structure,
not whether a written action is physically plausible or a character truly remains consistent.

Every step supplies `parameters`, even if it is `{}`. A model step references a logical model
requirement such as `assistant`; non-model steps cannot carry model or adapter selections.
The local Draw Things Qwen requirement permits 4B vision or 9B text only. Other declared runtimes
remain unqualified; a declaration never activates a loader automatically.

Typed values include edit IDs/instructions/preservation requirements, per-image observations,
clip frame allocations, endpoint descriptions/reuse references, and review items. Python
`validate_value(type_name, value)` checks their schemas. For `clip_plan`, it additionally checks
unique clip IDs, contiguous starts and an exact sum to `totalFrames`.

Clip timing uses **project frames**: a clip at `startFrame: 0` with `frameCount: 120` occupies
frames 0–119. Engine-native frame grids, generated audio duration, transitions, interpolation and
any finishing retiming must be resolved when a later workflow stage creates actual render jobs.
The planning contract does not pretend arbitrary project durations are native H3/LTX durations.
Endpoint `reuseFrom` uses `{ "clipID": "previous", "endpoint": "last" }`. Cross-document endpoint
coverage, backward-only reuse, and continuous-clip handoffs are enforced by `movie.check_plan@1`.
Durations round to the nearest project frame, with halves rounded up and rounding reported.
Frame counts are distributed evenly, without exceeding the preferred clip length rounded to project frames.

Legacy `movie.allocate_clips@1` (see [v1 example](movie-planning-v1.json)) assigns each clip only its relevant action sentences. Timed outlines such as
`**0:00–0:15**` followed by `Visual:` and `Audio:` lines are supported; timed sections must cover
the movie with no overlaps or gaps. Visual sentences are divided across the clips in that section,
in order. Standalone framing phrases such as “Medium shot.” stay attached to the next action.
Audio directions stay in the story outline and are not mistaken for visible endpoint actions.
For untimed outlines, action sentences are distributed in chronological order across the movie.
Sparse outlines are not padded with invented actions: repeated beats receive a review warning.

## User LoRA and QLoRA manifests

A workflow step can name an adapter by ID with `adapter`. Supply that adapter's manifest using
`--adapter` during validation. Its exact base `family`, `variant` and `revision` must match the
step's model requirement; omitting a required revision is not treated as compatible. The adapter
must also declare the selected runtime. The same ID cannot identify two supplied manifests.

An adapter manifest records:

| Field | Meaning |
| --- | --- |
| `purpose` | Prompt editing, movie planning, visual description, character, style or motion |
| `baseModel` | Exact model family, variant and revision; this is separate from its storage path |
| `weights` | Portable filename, declared byte size, SHA-256 and inference file format |
| `training` | `lora` or `qlora`, training-base precision, optional dataset ID/revision |
| `adaptation` | Rank, alpha and target modules |
| `inference` | Declared runtime IDs, default strength and vision-support qualification |
| `license` | SPDX expression or clearly identified custom license reference supplied by the author |

QLoRA describes training with a quantized base. It is not a universal inference file format.
`training.method: qlora` can accompany `weights.format: peft-safetensors`; the runtime still needs
an appropriate loader and compatible tensor mapping. Metadata never proves those are available.

**The Qwen helper does not currently load task adapters.** A structurally valid manifest reports
`executionStatus: not_implemented` and warns that weights and quality have not been checked.
This command intentionally does not open a declared weight file or verify its checksum; the
future importer must verify bytes before loading and qualify preserved vision independently.
No adapter or base weights are bundled with these examples. Keep weights in the user's existing
store; an execution binding will select them without copying a base model for each workflow.

Existing H3/LTX and Draw Things image/video LoRA controls are unchanged. Their accepted formats
and runtime tensor checks remain authoritative; this new task-adapter manifest is not yet an
import path into those controls.

## Run in Studio or headlessly

If your managed runtime predates workflow support, set up an updated runtime in Runtime Settings;
existing environments are preserved. The new runtime lock includes the validation dependencies.

In Studio, choose a built-in or **Import definition…**, fill in its inputs and locate the
required installed Qwen checkpoint. Add reference thumbnails to image inputs. **Run next**
executes one incomplete step; **Run remaining** continues the workflow. **Pause** cancels the
active local model call. Select a step to inspect its outputs, or **Regenerate selected** to
invalidate it and dependent steps. **Use proposal** returns staged text to the Prompt Assistant
for editing and explicit application; it does not alter clip settings or start a render.

The legacy movie planner's `story` and `clips` steps declare `requiresApproval: true`. New Studio
movie workflows default to **Create a movie**, with the guided creative-brief and review stages.
Execution stops
with `status: awaiting_approval` and `awaitingStep`. Select the step, review/edit its result and
choose **Approve step**, then **Run remaining**. Native editors expose character descriptions,
chronological actions, and individual clip action/start/end/location/continuity fields.
**Approve** on a clip locks that saved choice; **Unlock** permits edits or repair. **Repair**
reruns the chosen clip and rechecks dependent continuous clips. Independent cuts and unchanged
approved clips reuse their item cache. Incomplete item caches are sensitive to model/runtime changes;
completed reviewable steps use the content-preservation rule below.
Stale outputs remain readable but cannot be approved until recomputed. A step-level approval
must be renewed when a dependency changes, even if some individual clip approvals still apply.

Review mutations use the same exclusive lock and atomic checkpoint as execution. They require
`expectedRevision` from the latest `run.json`; stale UI/headless clients fail without writing.
They validate output types, IDs and frame layout before saving. Human edits do not invoke Qwen.
Successful review actions also append revision-linked human decision records in the same atomic
checkpoint. These preserve changes, approvals and model evidence without granting training consent.
Reference additions, removals and order can be saved with subject text. Exact helper token counts
and available preparation/load/prefill/decode timings are retained in the model-turn archive; missing
vision-stage timings are left unavailable. The helper rejects impossible input budgets before loading
weights, and non-retryable context/model errors require a changed task or model binding.
The CLI supports `--review /path/to/review-action.json`, for example:

```json
{"stepID":"clips","expectedRevision":"COPY_CURRENT_REVISION","action":"approve","itemID":"clip-1"}
```

Actions are `approve`, `unapprove`, `edit`, or `repair`. Omit `itemID` for whole-step approval
or unlock. `edit` supplies the complete typed `outputs` object; clip timing and cast cannot be
changed there—use movie inputs or the story step. `repair` requires `itemID` and accepts `instruction` (up to 2,000 characters) describing the desired correction; after the review
command, run the job again to execute it. CLI approval gates are never silently bypassed.
Studio asks what should change and performs the follow-up run automatically for **Repair…**. This uses a new instruction rather than simply replaying the same greedy model request.

Optional `fieldScope` on a `repair` mutation selects `action`, `states` (start/end), `location`
(location/continuity), `characters`, or `all`. A missing scope retains legacy whole-shot behavior.
The runner supplies the saved shot and copies only scoped fields from the response; ID, timing and
out-of-scope fields remain unchanged. Continuity and location constraints still validate the merged
shot. A one-field response is sufficient for a narrow repair. Human decision records retain the
scope and before/after evidence without granting training consent.
After a successful repair, its scope is consumed into a `lastRepair` record. Later continuity
refreshes use the current shot rather than replaying the old correction base. Failed or cancelled
repairs remain resumable; an explicit manual edit supersedes the pending repair.

Checkpoints retain the exact model request text, responses, reference IDs, model/helper file
fingerprints, image hashes and decoding settings for newly executed calls. Normal resume reuses
saved results. Fresh inference across hardware/runtime versions is not promised byte-identical.
The existing 2 MiB checkpoint and declared disk budgets still apply; no model/media copies or
unbounded checkpoint histories are created. Local model/helper fingerprints use size and mtime,
not a cryptographic hash of multi-gigabyte weights. Retain the installed model version when exact
provenance matters. A future artifact/image generation stage still needs its own qualification.

Studio saves small jobs under its Application Support `Workflows/Jobs` directory and states
under `Workflows/Runs`. **Resume last** or **Open job…** restores a job. **Export job…** writes
an executable local job including model/media bindings and the helper path. Exported jobs
reference files in place: they are not portable media bundles and contain no cloud credentials.
On another Mac, update file bindings and choose a new `runDirectory`.

The same job runs without Studio:

```bash
python scripts/run_studio_workflow.py /path/to/workflow-job.json
python scripts/run_studio_workflow.py /path/to/workflow-job.json --next
python scripts/run_studio_workflow.py /path/to/workflow-job.json --regenerate plan
```

The CLI emits JSONL progress and a final result. Exit status is 0 for completed/paused/awaiting approval,
1 for failure, and 130 for cancellation. SIGINT/SIGTERM cancel the owned helper and preserve
successful calls. Do not send these jobs to `render_headless.py`; that command handles render
recipes, while workflow jobs coordinate assistant/planning steps.

A minimal local job (replace placeholders with existing absolute paths):

```json
{
  "builtin": "staged-prompt-editing",
  "inputs": {
    "source": "A warrior in a cathedral.",
    "instructions": "Replace the warrior with a red fox. Move the scene to a snowy forest.",
    "images": []
  },
  "models": {"assistant": "/models/qwen_3.5_4b_i8x.ckpt"},
  "assets": {},
  "runtime": {"drawThingsHelperPath": "/app/WeeToddDrawThings"},
  "runDirectory": "/jobs/fox-edit",
  "maxTokens": 1024
}
```

Use `definition` for an inline portable definition or `definitionPath` for a local definition
file in place of `builtin`. For vision, put IDs such as `asset:character` in `inputs.images`
and their absolute image paths in `assets`. Model bindings use the logical names declared in
`models`. Built-ins require 4B because they include vision-capable operations, even with no
images selected. A custom text-only definition can declare and bind 9B.

## Inspect execution history

**Model turns…** is the full transcript inspector. New calls are recorded in a separate
`model-turns.sqlite` file, including exact system/input messages, response text, image IDs/bindings,
model/runtime metadata, settings, timings, errors and parse-validation outcomes. Cancelled calls
retain their request even if no response arrived. Cache reuse is a separate turn with zero model
execution time. Whole-step reuse creates an overview entry rather than a fabricated model turn.
Rejected responses survive cache clearing, and transcript rows survive overview retention trimming.
The database is read-only from Studio and messages load on selection; lists use 100-row pages.

SQLite's page limit caps the file at min(16 MiB, (maxWorkingBytes − 4 MiB) / 2), leaving room for
its rollback journal and the maximum atomic checkpoint write. This requires five artifact slots
for model calls. Response space is reserved before submitting a call. Exhaustion stops execution
with an explicit diagnostic rather than silently dropping turns. Old runs expose the messages
still in their checkpoint, with missing fields/order uncertainty disclosed. A parser result records
JSON/schema validation only; later semantic checks and human approval remain separate.


Studio's **Execution history…** panel shows each step's purpose and recorded timings, plus the
new bounded `executionHistory` records in `run.json`. Each activity records its step/operation,
reason, start time, duration, status, actual model-call count, reused-response count and step-level
retry count. Per-object corrective calls contribute to model-call counts even when the parent
step's own `calls` array is empty. Manual description/coverage reviews are separate activities.
Step failures and cancellations retain elapsed time. Resume creates explicit reused-step entries;
regeneration preserves recent historical activities independently of the active output cache.

The last 40 activities and 20 event details per activity are retained. `historyOmitted` and
`eventsOmitted` disclose truncation. Counters remain complete for each retained activity. Model
call events contain short task/system excerpts and request-text fingerprints, not another copy of
full prompts or output media. Fingerprints match request text/image IDs, not necessarily runtime
or image-file contents. `returned` means the model returned text; operation validation may still
reject it. Step retries count the outer operation retry loop; per-object repair calls are visible
in the request count and detail entries. An interrupted process can leave only partial timing.

Older checkpoints show their stored step durations and recursively counted saved responses, not
invented lifetime call counts. History is observational: it never participates in cache keys,
approval decisions or generation inputs. It stays inside the existing checkpoint size/disk limits.

## Execution and review limits

The runner validates definitions, actual input bounds, model variants, local files and output
port types. There is one weighted call at a time. A run-directory lock prevents simultaneous
writers. A step’s time limit covers its calls/retries in one invocation; a later resume has a
fresh time window. `maxStepExecutions` counts actual model calls and deterministic operations,
including retries. Its validation bound includes up to eight image descriptions and up to two
endpoint calls per clip. Continued clips reuse the first endpoint and need only one new call.
Changing the request establishes a new declared budget; repeated unchanged resumes retain
consumed calls. Raising a limit or starting a new run is explicit.

Completed reviewable steps retain saved outputs and approvals across runtime-only changes when
their content key matches. That key covers the saved step definition, resolved inputs, upstream
outputs and reference-image contents. A rebuilt helper or changed output-token limit alone does not
regenerate reviewed documents. Explicit regeneration and content changes retain normal invalidation.
Original execution provenance remains attached to reused results.

Within incomplete editing/endpoint steps, completed model calls may be reused from the execution
cache. That cache also depends on model/helper identity, implementation version and token settings.
Model identity uses path/size/modification time, including the tensor sidecar; it is not a
multi-gigabyte weight hash. Image references use SHA-256. Run files are local checkpoints, not signed
provenance documents. Raising a runtime limit does not grant approval or authorize regeneration.

Malformed structured responses stop safely or retry within the definition’s attempt bound,
with the validation error supplied to the next call. Truncated text is not silently accepted.
The built-in edit planner/reviewer permit one retry. The host constructs image observation IDs,
movie timing and endpoint links, avoiding model-written bookkeeping where possible.

Image observations are visible for review. An edit can set optional `useImages: true` when it
explicitly needs visual evidence; otherwise that evidence is excluded from its writing call.
This prevents an old reference subject from repeatedly overriding requested subject changes.
When the source draft is empty, observed image descriptions seed the initial draft. Each edit
still uses the **latest** draft. Creative and instruction-following quality remain
model-dependent. The Qwen review is advisory: a real test produced a false warning about a
correct one-sentence draft. Deterministic movie checks certify frame/link structure, not story
quality or identity preservation. The earlier whole-story-per-clip failure is fixed by scoping action text before endpoint writing.
Pacing checks flag repeated last-frame descriptions, an early fade/ending, and distinctive phrases
introduced before their assigned story action. These are conservative text heuristics: synonyms,
intentional holds and creative transitions still require human review. They do not certify semantic
correctness or generated motion. At most 1,000 review notes are returned, with a summary if more
exist. Inspect pacing and still-frame descriptions before generating assets.

## Next implementation milestones

1. Connect reviewed movie plans to existing image/video render jobs, with generated endpoint assets,
   timeline insertion, engine frame-grid/audio alignment and explicit generation controls.
2. Qualify Qwen task-adapter loading against exact installed bases, preserving vision. Adapter
   manifests already validate metadata; they do not enable a tensor loader or training pipeline.
3. Add user adapter import and reviewed dataset export with provenance, asset references and
   storage limits, then separately qualify LoRA/QLoRA training.


## Project subject proposals

`project.identify_subjects@1` produces `subject_list`. Subject IDs are assigned by the host from kind and normalized name; model-authored IDs are not used.
The internal model response uses `subject_selection_list` with one to three numbered `evidenceIDs`; the host validates selected IDs
against the active source section and copies the corresponding original passages to `evidence`.
Passages are at most 700 characters; model source windows are at most 5,000 UTF-8 bytes and the
workflow accepts at most 16 windows. There are three kind-specific passes and at most 48 calls per
attempt. Repeated subjects within a kind retain their first description and collect source evidence;
a differing description becomes an explicit review note. Cross-kind duplicate identities are
rejected. The final inventory is bounded to 24 proposals and remains subject to human review. Sections that
reach eight proposals add step warnings, retained as project review notes on import, because more
subjects may be missing.

Studio imports completed subject, story and clip outputs additively into its optional project
planning document. Stable source keys prevent repeated imports from duplicating existing records.
Explicit human approval of an inventory or individual subject carries into newly imported project
subjects. Model review reports do not grant approval; shot and reference approvals remain separate.
Reimport matches the workflow source and stable subject ID even after reclassification, preserving
existing project edits and references. Ambiguous legacy duplicates must be merged before reimport.
Project reference
images are ordinary Project Assets linked by ID; the source file is not copied. Planning JSON export
is separate from an executable workflow job. Automated sheet/endpoint generation and timeline
application have not been enabled by this milestone.


## Visual subject review

[subject-inventory-reviewed.json](subject-inventory-reviewed.json) runs
`project.identify_subjects@1`, `project.link_subjects@1`, `project.review_subjects@1`, and finally
`project.review_object_coverage@1`, then waits for human approval. Version 1.2.0 adds this final
coverage pass; previously saved definitions remain pinned.

Coverage requires bindings for `subjects: subject_list`, `brief: text`, and `library: object_catalog`.
The shipped workflow's library input defaults to `[]` when no catalog is supplied. It returns
`subjects: subject_list` with validated relationships, exact
`descriptionMentions` anchors and advisory `coverageReview` issues, missing objects and library
matches. Each anchor stores its target ID, exact phrase and zero-based occurrence alongside
`mentionSourceDescription`; changing the text invalidates its semantic anchors. The pass is limited
to 24 subjects and two attempts each, with resumable per-subject records. Independent valid entries
survive rejected proposals. Library candidates contain identity, kind, name, aliases, tags,
description, package/version, scope and definition revision; they never contain model weights or
copied media. Studio supplies at most 64 candidates and explicitly confirms reuse on project import.
The `review_object_coverage` review action also supports completed older checkpoints. Approved
objects and their dependency closure are preserved with separate proposals; unlock and rerun to
apply changes. The agent cannot approve records or silently create missing objects.
The older `subject-inventory.json` still loads existing checkpoints. Its subjects can use the same
manual review action without re-extracting the script.

Studio's **Add reference images…** binds existing files to portable asset IDs; **Review & improve
description** sends those IDs to both drafting and critique through local Qwen3.5 4B.
The `workflow-review` bridge action `review_description` takes `stepID`, `itemID`,
`expectedRevision`, and an optional `referenceAssets` list (maximum eight IDs from the job's
`assets` binding map). Save the job with those bindings when exporting/reopening the review.
This action preserves subject identity and explicit approvals, saves model call traces and elapsed
time, and invalidates dependent results. It never approves a subject. A failed/cancelled call can
change the checkpoint revision; reload `run.json` before retrying.

The optional `descriptionReview` report records criteria, proposed additions, image observations,
issues and the exact reviewed description. Human approval accepts the saved description independently of a failed or stale agent report;
that report remains visible and is never rewritten as an agent pass. Empty descriptions, stale
checkpoint revisions and incomplete subject steps remain blocked.
It is a model-assisted quality check, not proof of factual correctness.
