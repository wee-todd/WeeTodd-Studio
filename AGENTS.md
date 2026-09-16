# WeeTodd Studio agent guide

WeeTodd Studio is a standalone macOS creative app for Apple Silicon. The Swift app is the primary
product; ComfyUI nodes remain a maintained secondary interface to the shared renderer and media
utilities. Draw Things integration, native MLX inference, creative planning and reusable production
assets belong to the same product.

## Product direction

- Prioritize standalone-app usability, performance and complete creative workflows. ComfyUI is
  not a prerequisite for Studio; preserve node contracts and workflow compatibility as shared
  capabilities evolve.
- Treat Draw Things as a central integration and the preferred starting point for fast inference
  when it supports the requested model/task. Its connection helper can remain an optional runtime
  dependency. Distinguish local/self-hosted inference, Cloud API execution and native model reuse.
- Native inference provides additional models and advanced tasks, including supported A2V/V2V,
  conditioning and controls beyond the Draw Things integration. Current native engines are H3,
  LTX 2.3 and LTX 2.5. Consider additional models when they close a useful feature gap.
- Prefer sharing compatible installed weights in place, including supported Draw Things model
  files, before requiring another download/conversion. Compatibility is per component and task;
  model reuse does not imply Draw Things sampler parity or support for every native task.
- New Draw Things models should fit the existing discovery/interface where possible. Verify their
  capability mapping, settings and transport before claiming support; model discovery alone is
  insufficient. Keep model/backend limitations visible to users and retain useful native features.
- Product/repository name: WeeTodd Studio / `wee-todd/WeeTodd-Studio`. Keep existing package names,
  import modules, node IDs, document formats and saved local paths compatible unless a task
  explicitly includes a tested migration. Do not rename a user's checkout or model library as a
  side effect of the product rename.

## Scope boundary

- Supported scope includes Studio editing and creative planning, Draw Things image/video integration,
  shared model storage, native MLX engines, maintained ComfyUI nodes and relevant media utilities.
  New model integrations must serve the product direction and the assigned task.
- Keep native engines, local planning and optional remote execution behind their existing adapters.
  Do not expand an assigned change into unrelated application or account functionality.
- Independently implement and test native algorithms researched from third-party implementations.
  Optional runtime dependencies remain separate and retain their licenses and distribution limits.
- Never commit model weights, outputs, caches, tokens, credentials, or machine-specific paths.

## Development rules

- Keep node imports lightweight; load MLX weights only when a graph executes.
- Keep the H3 and LTX engines isolated behind separate adapters. Studio and headless jobs must
  reuse the shared renderer without importing ComfyUI or maintaining a second sampler.
- Before changing an LTX 2.5 loader, sampler, VAE, conditioning contract, or optimization default,
  compare current Lightricks LTX-2 releases, LTX-2.5 checkpoint files, and native ComfyUI changes
  against the baseline in `docs/reference/LTX25_MLX_INTEGRATION.md`. Update the baseline and OKF log
  when upstream changes. Do not transfer CUDA performance claims to MLX without measurement.
- Preserve synchronized audio and video as a single H3 generation contract.
- Keep model state process-local and explicitly unloadable.
- Default weighted stages to staged unloading: Qwen3-VL, transformer, video VAE, then audio VAE.
  Keep a component warm only through an explicit node control and report its resident state.
- Release the active component after success, failure, or cancellation when staged unloading is
  selected. Do not load the next weighted stage before the prior stage is releasable.
- Validate dimensions, duration, checkpoint paths, and task support before expensive work.
- Add a focused test for every node contract or engine behavior changed.
- Treat portable workflow paths and runtime-ready workflow paths as separate validation states.
- Before an expensive saved-workflow render, run `scripts/preflight_h3_workflow.py` with the saved
  API prompt and the active ComfyUI root. Execute that saved API prompt for the render.
- Keep local research, attribution, and knowledge-store material outside the tracked repository.
- In zsh commands, do not assign `path` or `PATH`; zsh ties `path` to the executable search path.
  Use a task-specific variable name such as `ltx_source_rel`.
- Do not copy incompatible or unlicensed third-party code into Apache-2.0 files.
- Use `.agents/skills/wee-todd-h3-mlx/SKILL.md` for H3 implementation work.
- Before changing Python interpreters, venvs, MLX builds, pip packages, or dependencies, use
  `.agents/skills/python-environment-preflight/SKILL.md`. Run its preflight before mutation.
- Before risky edits, use `python3 scripts/create_source_backup.py --name <short-name>`.
  Do not pass the repository root to a generic recursive snapshot tool: ignored local state may be
  many gigabytes. The source-backup command uses Git's tracked/unignored file set, applies the
  protected prefixes in `.source-backupignore`, writes outside the checkout by default, and rejects
  unexpectedly large inputs before copying.
- Before every commit, use `.agents/skills/readme-workflow-commit-gate/SKILL.md`. Audit the complete
  `README.md`, `studio/README.md`, current `STATUS.md` claims, the Studio workflow guide, and every
  shipped ComfyUI UI/API and Studio workflow definition, then record review for the exact staged
  snapshot. Do not commit when the gate or its local pre-commit hook fails.
- Keep the generated README node catalog synchronized with every registered node and its current
  behavior. Run `python scripts/update_readme_node_catalog.py --check` before every commit. Update
  the node note and maturity status when a node contract changes.

## Skill routing and project overrides

Read the skill for the work being performed; a shared product keyword alone is not a trigger.
Use the session's actual tool schemas and permission rules over examples in installed plugins.
Reuse an already approved design and keep tool waits short enough for the session's progress updates.

| Work | Local skill or contract |
| --- | --- |
| Native H3 engine, shared execution or node contracts | `.agents/skills/wee-todd-h3-mlx/SKILL.md` |
| Native LTX loaders, sampling, conditioning or memory | `.agents/skills/wee-todd-ltx-mlx/SKILL.md` |
| Studio UI, project persistence or creative workflow review | `.agents/skills/wee-todd-studio/SKILL.md` |
| Draw Things remote adapter, transport, allowance or resume | `.agents/skills/wee-todd-drawthings-adapter/SKILL.md` |
| H3 or LTX 2.5 prompt authoring | The corresponding `h3-video-prompting` or `ltx25-video-prompting` skill |
| Python environment or dependency mutation | `python-environment-preflight` |
| Commit creation, amendment or merge | `readme-workflow-commit-gate` |
| Local knowledge-store edits | `okf-knowledge` |

Draw Things CLI generation skills apply when running that CLI, not when editing the remote adapter.
Repository backup, documentation-storage and commit rules supersede generic skill defaults:
use `scripts/create_source_backup.py`, keep plans/research local, and do not force-add ignored plans.
Do not edit vendor plugin caches or global invocation settings as a side effect of project work.
These local skills are intentionally untracked. In a linked worktree, read them from the installed
checkout; the hook installer records the local gate's location. A fresh clone uses the tracked
validation entry point below and needs a separately installed local skill for manual commit review.

## Validation

Run core plus the profiles affected by the change. Profiles combine in one invocation and share
one interpreter preflight, test selection and lint command. They do not install dependencies,
run model generations, replace an app bundle or record manual review.

```bash
python scripts/validate_project.py --profile core
# Add --profile studio, --profile workflows, and/or --profile remote when affected.
```

Core covers ordinary top-level native/shared Python tests, compilation, README/catalog checks and
portable ComfyUI workflows. Workflows adds Studio definition schemas and workflow execution/review
tests. Studio adds its Python bridge/packaging tests and Swift tests; remote adds Python adapter
tests and the Draw Things Swift package. `--list` prints the exact checks without running them.

Only after the complete manual review, record it against the final staged snapshot:

```bash
./.venv/bin/python .agents/skills/readme-workflow-commit-gate/scripts/audit.py \
  --project . --record-review \
  --confirm-readme-reviewed --confirm-public-docs-reviewed --confirm-workflows-reviewed
```

External-reference packing parity, algorithm search, full checkpoints and real generations remain
separate optional qualification. State clearly which were not run. A push-only operation does not
create a new staged review; report existing committed-tree validation or run needed read-only checks.

For Studio changes, use the Studio profile and run
`python3 scripts/build_studio_app.py --configuration release` after saving and quitting that app.
For validation while the normal app remains open, use an isolated bundle instead:
`python3 scripts/build_studio_app.py --configuration release --output /tmp/WeeTodd-Validation.app --signing-identity -`.
Shipped runtime setup must use `scripts/preflight_python_environment.py`; it cannot depend on ignored
local skills.
Keep Studio projects, exported jobs, collected media, and app-managed runtime data out of commits.
