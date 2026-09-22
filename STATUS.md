# WeeTodd Studio implementation status

Local assistant sessions and character isolation 2026-09-21 (experimental): related Qwen3.5 4B
calls now share app-owned decoder/vision weights and a content-hashed image cache within one job.
Execution remains serial; attention, convolution and recurrent state start fresh for every request.
The cache holds at most four image sets / 128 MiB of prepared tensors and encoded features,
excluding model weights. Cancellation, failure, model changes and the return to human review unload
the worker; the shared inference lease survives abrupt parent exit until its worker exits.
The 9B text route remains one-shot. Apple Vision isolates one unambiguous face-associated foreground
subject on white before preparing 512-pixel overview/head/torso inputs, with an explicit bounded
fallback when isolation is unavailable. Originals remain intact and source revisions are rechecked.
Character values request supported descriptive clauses while retaining separate evidence; partial
schema repairs preserve valid fields. Physical measurements remain manual or authored-text inputs.
Optional visual records require an observed inventory item, including a single feature slot.
These controls improve preparation and validation, but proposals still require visual review.

Installed 8-bit 4B qualification on M3 Ultra / 256 GiB used the same isolated character inputs and
18-call extraction job: one-shot took 128.8 seconds, session reuse took 84.9 seconds (34% less time).
Sampled helper RSS maxima were 4195 and 3858 MiB respectively; these are not peak physical-footprint
measurements or cross-device memory guarantees. The session loaded each model component once,
encoded three image sets and served 15 image-cache hits, retaining 27.8 MiB of image data. Three
mixed long-text/image/short-text/repeated-image sequences matched the one-shot answers exactly;
median wall time was 13.04 versus 6.43 seconds. The full extraction produced 69 versus 72 proposals,
so these timings do not establish exact full-job output parity. Nine installed-model runtime tests
passed, including long-prefill initialization, fresh request state, cache invalidation, cancellation
and small output budgets. No new image-generation or cross-device quality qualification is claimed.

Structured Character Director 2026-09-20 (experimental): a dedicated movie-independent window and
shared character-reference editor now use typed appearance fields, a deterministic thirteen-section
prompt, forensic Qwen proposals, versioned style presets and separate portable documents. The local
Krea 2 Turbo recipe requests a 1920×1088 four-panel row; Vision proposes actual panel crops for review,
then serial FLUX.2 klein 9B refinement produces a 3840×2176 assembly. Separate head replacement
runs BFS at native panel size before exact 2× Lanczos preparation for HighResolution9B detail.
Reference-head masking retains RGBA and white-matted derivatives. Progress, previews, cancellation,
content-hashed completed-panel reuse and explicit uncertain-submission retry are wired. Assembly
provenance retains panel settings and reference hashes. Follow-up qualification on 2026-09-21:
local Krea generation and the four-panel
Klein/BFS/detail pass have completed with live previews and verified completed-panel reuse. Vision
preserves separate foreground instances and combines body detections to handle unequal panel widths.
New documents now default to separate eight-step BFS-only and HighResolution9B-only passes,
preferring an unambiguous installed 9B KV model; saved recipe settings are preserved. New prompt presets
separate trigger-only BFS from concise HighResolution quality wording. Legacy prompt contexts and
explicit strengths retain their behavior; strength controls show percentages. Qwen wardrobe analysis
uses a bounded torso/lap detail, an initial item inventory and literal condition-evidence checks,
with surface targets tied to observed garments. These checks do not establish visual correctness. A complete
8-bit KV run with native-size BFS before 2× detail took 600.5 seconds with 40 live preview revisions
on an M3 Ultra / 256 GB. The close-up still invented top hair and exaggerated skin texture.
A subsequent minimal-prompt comparison at 0.80 detail strength improved shirt/denim consistency,
but BFS turned the back-view head toward the camera and the close-up still exaggerated skin contrast
and changed lens tint. The back-view error exists before upscaling; this recipe has not passed visual
acceptance. The generic detail-strength default remains 1.0 pending stronger quality evidence.
Qwen character extraction includes bounded Vision head/scalp details and dedicated visible skin texture fields. Head consistency, surface realism and composition
remain experimental; the earlier combined-pass runs do not establish general performance
or peak-memory guarantees. See the
[Character Director guide](studio/README.md#character-director) for current controls and limits.

Native Qwen-Image-2.1 2026-09-20 (experimental): the existing image workspace now selects an
app-owned MLX image engine with up to ten total ordered references, 8-bit model preparation,
approximate live previews, stage/step/layer progress, cancellation, staged unloading and shared
weighted-job admission. Native settings and excess cards survive backend switches. Reference/Ripple
image editing and v4 headless jobs share the same validated renderer. Draw Things retains its
existing transport and limits; future Qwen 2.1 remote support is not claimed.

Official revision `b3179ad355be050328e483a9dfdd9e60cd62adfa` was prepared with its vision tower intact.
Tiny FP32 comparisons against pinned upstream implementations reached maximum absolute errors
of 5.22e-8 for encoder features, 7.38e-7 for transformer trajectories and 1.44e-6 for VAE output.
This does not establish full-checkpoint 8-bit parity. A 512×512, 40-step red-to-blue teapot edit
completed in 51.2 s including initial verification, with 8.82 GiB peak MLX allocation and 14 previews.
A ten-reference 1024×1024, 40-step badge render took 390.4 s inside the renderer, with 34.74 GiB
peak MLX allocation and 40 previews. Both released MLX active allocations to about 1 KiB and
reported no swap growth on the Apple M3 Ultra with 256 GiB unified memory. Process footprint and MLX
allocation are distinct measurements; these single cases are not minimum-RAM recommendations.
All ten badge numbers appeared in order, but some colors changed. Broad identity/text/alpha quality,
lower-memory hardware qualification and full-checkpoint numerical parity remain open. Native LoRAs,
tiled VAE, persistent warm models and dynamic upstream-to-native image bindings are not included.
The checkpoint's noncommercial Qwen Research License requires separate commercial permission.
A follow-up allocator-cache limit during VAE execution preserved a byte-identical 512-pixel test
image and reduced its post-render physical footprint from about 7.0 to 1.2 GiB. A 2048-square
four-step run completed with 45.5 GiB peak active MLX allocation and 40.3 GiB physical footprint;
it qualifies allocation/decode behavior, not final image quality. Memory admission now reserves
16 KiB per output/reference pixel for the untiled VAE working set, provisionally above observed
512–2048 peaks. Ten sequential completed/cancelled jobs returned to 1,068 active MLX bytes each;
five cancellations at stage boundaries released within 0.13 s. Arbitrary mid-kernel cancellation
may take longer and retains the worker termination fallback.

See [native image setup](studio/README.md#native-qwen-image-21--experimental).


LTX Ripple editing 2026-09-19: Studio adds a general clip inspector and a dedicated Ripple
Director with a resizable source preview, frame scrubbing and clickable reference thumbnails.
Frame zero is required; up to eight additional distinct references can be added. The frame editor
opens Draw Things with the extracted source image on its canvas or accepts an existing edited
image. Thumbnail context menus delete additional references while retaining the first-frame slot.
The shared local LTX 2.5 renderer executes the verified v11 IC-LoRA with eight full-resolution
Euler steps. Its editable strength defaults to 1.35, and the editable prompt starts with the
author's recommended propagation instruction. Adapter/component locations are runtime settings.

Source audio is preserved by default, with silent inputs and silent output supported. Source
frame cadence, fractional trim positions, display rotation and pixel aspect are validated;
variable-frame-rate sources are rejected before inference. New takes require explicit application,
retain frozen edited images and receipts, and verify the source identity when replayed. Document
session checks protect late image and video results. See [Ripple Director](studio/README.md#ltx-25-ripple-director).

Qualification: one author-paired source/edited-first-frame example rendered five seconds at
1152×768, 24 fps, seed 42 and strength 1.35. Sampled frames showed the white-fur edit throughout
the kitten's movement; final publication was verified as 120 frames and 5.000 seconds. This source
was silent. Separate real-media tests verify source-audio replacement, leading offsets, silence,
exact frame extraction and trim. Additional timed references are an experimental Studio extension;
nine-reference visual quality and broad edit consistency remain unqualified. Core/Studio validation
passed 2,698 Python tests (three skips) and 505 Swift tests (three skips, zero failures), and release
packaging and isolated Director interactions were checked.

Fish voice direction 2026-09-19: synthetic and sampled voices now expose pitch, pace, timbre,
accent and a short free-form voice description, with per-character settings for conversations.
These compile into visible Fish instructions without changing the spoken script. Direction is
guidance rather than an identity or accent guarantee; reference speech remains the strongest
available identity control. Qwen keeps its model-specific preset/instruction controls.

Native voice and multitrack audio 2026-09-19: Studio's Voice workspace adds independently
implemented local MLX Fish S2 Pro (qualified community 8-bit/BF16 layouts), Qwen3-TTS 12Hz Base
(1.7B/0.6B 8-bit), and Qwen3-TTS CustomVoice (1.7B 8-bit). Fish and Qwen Base accept sampled audio
with a transcript; Qwen Base also supports audio-only speaker identity. Fish supports synthetic
speech. CustomVoice provides nine preset voices and delivery instructions, without sample cloning.
Speech model locations are configured once in Runtime settings; Voice exposes Family → Installed
model, remembering each family's last selection. Legacy draft locations migrate into Runtime
settings; completed takes retain their resolved request and native-rate audio/artifacts.

Fish scripts have clickable delivery tags and reviewed local Qwen3.5 auto-tag suggestions that
preserve spoken words. Both families support sequential multi-character conversations, per-speaker
references or preset voices, per-line performance overrides and pauses. Qwen Base delivery depends
on expressive references; CustomVoice accepts text instructions. Line artifacts and sample-exact
timing remain available. VoiceDesign, speech fine-tuning and speech LoRAs are not supported.

Separate clip-anchored Voice and Music tracks share gain, pan/balance, independent fades, explicit
equal-power crossfades, optional voice-driven music ducking and per-track stereo reverb.
Room/Chamber/Hall/Plate presets expose amount/decay plus advanced tone/pre-delay; tails cross region
boundaries and gaps, ending at the movie boundary. Music ducking also lowers its reverb tail.
One mixer prepares preview, export and Voice/Music/combined drivers for independent native clips.
Older projects preserve their legacy mix policy and open with reverb off. Existing scene and
Director workflows are not rewritten. See [Voice and audio mixing](studio/README.md#native-voice-and-audio-mixing)
for model setup, reference preparation, controls and the separate model licenses.

Real speech qualification: both Fish formats and both Qwen Base sizes generated reference speech.
Qwen's two reference modes and Fish synthetic speech produced seven checked takes; a local English
recognizer recovered the exact target sentence from each. Later Fish tagged and Qwen Base reference
conversations completed, as did a two-speaker CustomVoice conversation from the app. CustomVoice
neutral/excited instructions changed generated codes and duration for the same script/voice/seed.
Recognition recovered the Qwen wording; the whispered Fish line had approximate recognition.
These checks establish execution and short English wording, not subjective voice matching,
emotional quality or general multilingual accuracy.

Native H3 and LTX 2.5 completed combined-driver tests, initially at 384×256 and subsequently at
1280×768. The later exports were exactly 5.000 seconds, 24 fps, stereo 48 kHz AAC. H3 used a full
Larry Turbo v4 adapter at strength 1 with eight evaluations through the existing headless Ref2VA
path; Studio's ordinary Turbo picker retains its task restrictions. LTX used distilled Q8 8+3
sampling with its two-stage upscale. Sampled frames were inspected; measured lip-sync quality
remains unqualified. Voice-only/Music-only driver routing has automated tests; real video tests
used the combined mode.

A four-track, ten-second cached-source mix rebuilt in roughly 350–410 ms on the qualification Mac,
before UI debounce/player replacement. Reverb DSP alone processed 60 seconds of stereo synthetic
audio in approximately 0.083 seconds (Room) or 0.373 seconds (six-second Hall decay), after imports,
with at most about 40 MiB temporary DSP allocation. These exclude media decoding/encoding and are
not long-movie latency guarantees. Preparation uses disk-backed continuous PCM, one-hour input
limits, bounded disposable caches and chunked output verification; effects require mix preparation.
Model-free regressions cover stereo/antiphase preservation, damping, convolution continuity,
bypass, region splits, driver crop parity, ducked tails, cancellation, stale inputs and legacy policy.
The isolated release app exercised presets, advanced controls, bypass retention, saving and playback.
Listening approval, full upstream checkpoint parity and low-memory hardware remain unqualified.

Timeline and export reliability 2026-09-18: native playback now quantizes shared composition
boundaries consistently, preventing black video from sub-frame instruction gaps. Playback ticks
update the playhead without rebuilding the full editor; FF/LF controls seek in movie time, including
transition offsets. Reopening a project refreshes generation descriptions and revalidates saved
takes before production reuse. Movie assembly trims AAC packet padding before concatenation so
cuts retain their planned timing. Fractional-cut, decoded-pixel and frame-order regressions cover
these fixes; live long-movie playback and seeking were also checked.

Lyric-assisted verification 2026-09-17: supplied lyrics now guide short phrase-boundary repairs
and acoustically checked corrections between matching context words. Studio retains raw recognition
and separate lyric-assisted wording, with per-word agreement/assistance/unresolved labels. Missing
or unsupported lyrics remain untimed and are not inserted into the transcript. Acoustic caches
remain reusable when lyrics change. In a small three-excerpt annotated English check (210 words),
word-edit errors dropped from 192 to 177 with the original mix and 166 to 157 after isolation;
mixed-audio words timed within 200 ms rose from 52 to 54, with no additional >1 s onset errors.
The generated song still supported 61/174 words; this is a conservative verification improvement,
not reliable automatic singing alignment. Cached-evidence comparison added roughly 1–4 ms per
excerpt and about 4 ms for that generated song on the test machine.

Resumable production 2026-09-17: Movie → Produce movie and the approved Shot List handoff capture
an immutable edit, render native units serially, prepare predecessor-dependent continuity in order,
and assemble through the existing movie mixer. Complete LTX2.5 scenes are one unit. Local retries
are bounded to 0–3; pause/resume retains verified takes and scene checkpoints. Final publication has
a recoverable journal. Applying takes verifies artifact/source receipts plus the unchanged project
and execution context; prior versions and Music track data remain intact. Self-hosted Draw Things
is explicit with no uncertain resubmission; Cloud API shots retain individual cost confirmation.
Real stereo assembly, failure/cancellation/resume and identity regression tests cover orchestration.

Native audio analysis 2026-09-17: independent MLX Beat This small0 and wav2vec2-base-960h
provide learned beat/downbeat events and English acoustic word evidence. Music-video intake
adds model setup, analysis preview, word/line support and omission/repetition flags, editable
frame-snapped cut markers and locks. Optional native UMX-HQ vocal isolation feeds only word
evidence; original audio remains the beat-analysis source and movie soundtrack. Stem and acoustic
caches are independent of supplied lyrics. Reviewed lyric cuts use explicit source seconds.
Song sections/repetition groups and vocal gaps are provisional programmatic suggestions.

Qualification: model numerical parity, all 16 words of a known spoken sentence aligned, inserted
words/absent lines left untimed. A 159.8-second song took 2.124 seconds cold and 0.184 seconds
cached on the test machine. Beat stage detected 283 beats/71 downbeats. On the same song with
its recovered generation-request lyrics, isolated vocals increased acoustic word coverage from
29/174 to 61/174; neither mode fully timed a line. Full-service timings were 2.09s mixed and 4.38s
isolated on the test machine. This is agreement/support against intended text, without independent
performed-lyric transcription or hand-timed boundaries. Sung timing remains unreliable; no claim
of accurate automatic song lyrics or semantic chorus detection.

Music-video planning and audio-driven scenes 2026-09-16: the new Movie action accepts songs,
supplied/unknown/instrumental lyrics, reference images and reusable production objects. Native
DSP onset/dynamics analysis is cached; bounded frame planning replaces a user-selected fixed clip
length. Shot List combining retains original shots and frame references, and approved shots can
be added to the timeline with the original Music track. LTX 2.5 continuous scenes now support
one shared, contiguous source-audio interval with endpoint/keyframe images. Both stages freeze
source audio; checkpoint resumes include source identity. The native chain intermediate uses ALAC
(up to 24-bit PCM); Studio/headless scene delivery restores the original interval as PCM32.

Real qualification: two shots/two image anchors, four seconds, 384×256/24 FPS, 43.8-second render
and 3.4-second resume. The source interval matched all 192,000 stereo samples exactly, and macOS
AVFoundation decoded it successfully. Scene members require the eight-frame grid and existing
2–6-member/30-second limits. English word alignment and learned beat/downbeat analysis are now available as described above;
singing accuracy, semantic chorus detection and music-video lip-sync quality remain unqualified. Workflow planning retains human reviews and
supports ordinary clip generation or the resumable production handoff after timeline application.

Native YuE2 music 2026-09-16: Studio's Music workspace uses an independently implemented MLX
score/semantic/acoustic/stereo-decoder engine, shared with headless music commands. No external
YuE inference package is installed. Guided musical tags, lyrics/instrumental mode, seed and quality
presets sit alongside explicit advanced samplers, ABC composition, guidance and precision controls.
Takes retain verified tokens/noise/latents and checkpoint content provenance for acoustic resynthesis
and decoder-only replay. Staged unloading covers success, failure and cancellation, including
weights retained by exception frames. PCM export applies global gain only when needed to avoid
clipping. The verified optional download uses the merged npario 8-bit layout; split layouts are
rejected. Weight licensing remains CC BY-NC 4.0, separate from the original engine code.

Generated music can be auditioned and placed on a stereo Music track under source audio, or
prepared as an exact, nonzero-offset excerpt for an existing native A2V clip. Default placement
avoids source-replacement tracks. Async results protect changed documents/drafts; Collect Media
preserves stage artifacts. Guided music-video planning now uses the existing reviewed Director and Shot List workflow; see the music-video qualification above.

Qualification: a real 8-bit, 32-step song ended naturally at 159.8 seconds, generated in 123.2
seconds on M3 Ultra / 256 GiB (approximately 45.2 GB process peak footprint). This single run
predates the final integrity-hashing overhead and is not a general hardware benchmark. Verified
decoder-only replay took 4.26 seconds including 2.14 seconds of source/checkpoint hashing and
2.03 seconds of decoding. A three-second excerpt drove a real LTX 2.5 A2V render; output was
384×256 at 24 FPS with stereo audio. A separate additive Music-track export retained the main
audio and stereo separation; comparison accounted for the existing limiter's 239-sample lookahead.
Native app checks exercised model inspection, generation, truncation labels, decoder replay and
Music-track placement in an isolated app/data directory. Core + Studio validation and release
packaging passed, with focused final regressions for the new controls. M5 attention, lower-memory
hardware, real BF16/4-bit generations and full-model PyTorch parity remain unqualified; measured
audio validity is not a claim of subjective musical quality acceptance.

LoRA library 2026-09-16: Runtime Settings supports multiple linked LoRA folders, enabled/subfolder
controls, an optional training-model fallback and a default app-managed location. Header-only scans
deduplicate linked files, cache unchanged metadata and report missing drives or unsupported adapters.
One searchable library shows local files and Draw Things catalogs with explicit source/compatibility
labels. Clip and image inspectors show applied stacks; Draw Things group management moves into
the library. Image edits target the draft, and folder preferences do not invalidate render inputs.
Folder discovery does not mutate saved clips or groups. Draw Things stores remain connection-owned;
native use requires supported SafeTensors adapters. No conversion/upload or new sampler is introduced.

Reference inputs 2026-09-16: Studio exposes distinct appearance/story, motion/composition and
audio purposes. Local movie preparation creates visible Ingredients sheets or Canny guides with
verified cache reuse. LTX 2.3 setup includes dedicated Ingredients/Union adapters, and automatic
control selection matches the guide family. Draw Things H3 Ref2VA maps ordered image references
through the separate helper; movie/audio references and LTX IC-LoRAs remain native-only in this
integration. Original assets and incompatible attachments are preserved. New software checks use
synthetic media and a local transport fixture; no new real-model visual qualification is claimed.

Timeline playback 2026-09-16: the viewport now plays all timeline clips using movie time, with
click/drag seeking on the ruler and a draggable playhead triangle and vertical line. Blank-area
ruler clicks are ignored; handle drags clamp to the timeline. Scrubbing preserves the previous
playing/paused state and does not change inspector selection. Split uses the shot under the playhead;
keyframe attachment times remain local to the selected shot.

Native playback references trimmed source ranges in an AVFoundation composition, with bounded
preview dimensions, asynchronous metadata loading, still/unfinished-shot timing and active audio
regions. The editing view previews transitions as cuts at incoming shot starts; rendered movie
preview remains the check for actual transitions and the finishing mix. No preview movie needs
encoding for ordinary playback. New tests decode colored source ranges, exercise real playback
across cuts and empty intervals, and cover scrubbing, cancellation, frame offsets and audio fades.
Core/Studio validation passed 2,149 Python tests and 322 Swift tests (two optional skips), plus
release packaging. Live Studio verified ruler clicks, dragging both parts of the playhead,
ignored out-of-bounds clicks and a complete six-shot playback to the edited 28.75-second ending.
Existing movie content was preserved. No model generation or checkpoint parity was run.

Release validation 2026-09-16: the combined core, Studio, workflow and remote profiles passed
2,815 Python tests, 308 Studio Swift tests (two optional skips) and 41 Draw Things transport Swift
tests, plus lint, public-document links, the generated node catalog and portable workflow checks.
The README now introduces the shared continuity controls alongside setup; the Studio guide adds
model-switch, saved-context and boundary-flash troubleshooting. The release app was built and its
H3/LTX 2.5 connection menus verified during the continuity-naming check below. This final audit did
not rerun inference, full checkpoint parity or low-memory hardware qualification.

Continuity naming 2026-09-16: the native connection picker uses **Continue scene** for H3,
LTX 2.3 and LTX 2.5. Help text distinguishes continuation from an accepted take from LTX 2.5's
grouped render. LTX 2.5's separate source-video extension is **Extend previous take**. Experimental
status remains in the help text. Saved mode IDs, render routing and model-switch behavior are
unchanged; this is a presentation update, not cross-model continuation support.
Core/Studio validation passed 2,149 Python tests and 308 Swift tests (two optional skips), plus
release packaging. The rebuilt app's H3 and LTX 2.5 menus both verified **Continue scene**;
project settings were preserved. No inference or full checkpoint parity was run for the rename.

Model-switch scene repair 2026-09-16: changing a continuous-scene member to H3, LTX 2.3 or
Draw Things now separates that shot in the same undoable edit. Remaining LTX 2.5 groups retain
their effective shared sound and boundary-image policy. Images, takes and timeline ranges stay
intact. Saved incompatible native scene connections offer **Separate this shot**, and the picker
explicitly identifies the LTX 2.5-only scene route. Ordinary per-clip continuity is unchanged.
Regression tests cover every shot position, provider restoration, Undo/Redo, saved invalid groups
and unrelated parameter edits. Core/Studio validation passed 2,149 Python tests and 308 Swift tests
(two optional skips), plus lint/docs/catalog/workflows and release packaging. Live Studio repair
cleared the H3 shot's error and retained the preceding five-shot LTX scene. No inference was run
for this UI/state fix; native sampling and model support are unchanged.

LTX 2.5 strobe correction 2026-09-16: the dark pulses at native scene joins were already
present inside sampled continuation windows. Two interacting causes were reproduced: reusing a
sampled terminal video latent as interior history, and applying overlap images again alongside the
history they had already influenced. Both sampling stages now reuse interior history and regenerate
the terminal overlap slot. Studio's default **Automatic** boundary guidance applies each image once
at its requested strength; later windows inherit its influence through motion history. **Strict**
retains repeated overlap guidance. Preparation exposes image ownership without changing attachments,
timestamps or requested strengths. A separate fractional-mask error in noise-free Euler is also fixed.

A complete rerender of the original six-shot scene retained prompts, seeds, all 12 images and native
8+3 sampling. The five original join dips measured 6.91%, 10.39%, 29.72%, 32.80% and 29.02% below
both surrounding brightness medians; all five were absent by that metric after the correction.
Dense frame inspection confirmed removal of the dark pulses without added timeline crossfades or
output brightness correction. Natural motion and gradual exposure variation remain. Export fully
decoded at 30.000 seconds, 720 frames, 24 fps, 768×448 with synchronized stereo 48 kHz audio; the
runtime unloaded. Generation took 700.66 seconds (11:41) with concurrent validation/build activity,
so this is not an isolated speed benchmark or a universal continuity guarantee.

Validation passed 2,149 Python tests plus lint/docs/catalog/workflows and 303 Swift tests (two
optional skips). Release packaging passed and the updated app verified all boundary routing during
scene preparation. Existing user takes and crossfade edits were preserved. Full upstream checkpoint
parity, low-memory hardware and combined MSR scene generation remain unqualified.

Continuous scenes 2026-09-16: Studio now sends a connected group of two to six compatible native
LTX 2.5 shots (up to 30 seconds) through the existing audiovisual latent chain. Variable windows,
per-shot seeds, global first/last/timed images, shared sound, validated window checkpoints, grouped
job export and atomic whole-scene review are implemented. Frame-match conversion can freeze the
previous take's visible ending as an explicit image. Changed documents or inputs cannot accept a
stale candidate; retries preserve earlier candidate files. Preparation flags distinct image files
on consecutive boundary frames. This initial milestone excluded MSR/reference, audio drivers and
control adapters, including control adapters hidden in ordinary LoRA slots. The later music-video
increment above adds shared source audio; MSR/reference and control adapters remain excluded.

A front-end Rill qualification rendered six five-second shots as one scene and accepted all six
source ranges together. Export fully decoded: 30.000 seconds, 720 frames at 24 fps, 768×448,
stereo 48 kHz audio. Models unloaded after publication. The initial full generation took 12:57,
versus 8:55 summed across the earlier independent frame-matched clips; overlapping sampling makes
these different workloads. Story and identity persisted, but framing/appearance changes remain at
several joins. Inspection found those changes already in incoming sampled windows, with distinct
full-strength LF/FF images one frame apart. The controlled conditioning ablations and strobe correction are recorded above. This initial
result did not establish seamless output or music suppression.

That run exposed an unbounded full-volume convolutional-VAE decode: peak MLX allocation was
29.43 GB (27.41 GiB). Staged long scenes now use the existing temporal decoder with bounded
80-frame tiles and 24-frame overlap, then release it before audio decoding. A real decode-only
check using the same saved latents took 102.46 seconds and peaked at 6.87 GB (6.40 GiB) of MLX
allocation. It retained all 720 delivered frames and exact audio duration. Mean frame SSIM against
the full-volume output was 0.9873, with final-frame SSIM 0.9811: visually similar, not bit-identical.
This measures decode memory, not a new complete-generation peak. Ordinary short-clip and diffusion
VAE paths retain their existing behavior. Low-memory hardware and combined MSR scene generation
have not been qualified. Audio timing is verified; listening feedback remains pending.

Final core/Studio validation passed 2,113 Python tests and 302 Swift tests (two optional skips),
plus lint, documentation, catalog and workflow checks. Release packaging passed after saving and
quitting the app. The rebuilt app reopened the saved scene and verified the whole-scene duration
and boundary-image warnings during preparation. Full upstream checkpoint parity was not run.

Fix 2026-09-16: full-movie preview now compares the captured project value and document session,
not independently encoded JSON bytes. Unordered JSON keys could falsely report an edit in an
unchanged six-shot movie. Regression checks preserve rejection of real edits and reopened documents.
Validation passed 274 Swift tests (two optional skips), 55 Python bridge/packaging/README checks,
catalog and release packaging. Live Studio preview of the complete 30-second LTX 2.5 movie passed.

UI and render qualification 2026-09-16: generation now starts with **WeeTodd (local)** or
**Draw Things**. Local users choose H3, LTX 2.3 or LTX 2.5 and adjust task, sampling and LoRA
controls directly. Compatible components are selected automatically; custom recipes and execution
presets remain in Advanced generation. Returning to a provider restores its saved model settings.
Prompt review identifies the effective continuity source, excludes disabled LoRAs and distinguishes
media inputs from active adapters. LTX motion continuation is labelled as video extension.

A Studio-front-end exercise recreated the six-shot Rill movie with local H3 Q8 FL2VA, the
LightX2V Turbo adapter at strength 1, four evaluations, first/last frames and 22-frame synchronized
motion context. The exported movie decoded successfully: 30.000 seconds, 720 frames at 24 fps,
768×448, stereo 48 kHz audio. Each weighted stage unloaded after generation. Visual review found
recognizable action and scene continuity, with some mid-shot exposure variation still visible.

On the tested 256 GiB M3 Ultra, the first shot took 8:59 with lower-memory paging and 4:06 with
larger-workspace paging; sampling took 7:23 and 3:15 respectively. The larger-workspace take was
not byte-identical (whole-video SSIM 0.9741). Other continuing shots took 5:19–5:58. These are
single-scene observations, not Draw Things parity or a low-memory hardware qualification.
Total time also benefited from a new bounded model-checksum cache; warm continuation identity
verification took 0.35 seconds and matched the original full-hash identity. Payloads and manifests
remain fully checked. H3 prompt assembly now preserves native prompts with leading Picture
alignment clauses and shifts continuation anchors into the sampled window. Accepted frame-rounded
durations no longer immediately mark the finished take stale; document/result races remain guarded.

LTX 2.5 comparison on the same Approach shot completed through Studio with Q8 distilled 8+3
evaluations and no H3 adapter: frame matching plus last-frame guidance took 1:38; motion/audio
continuation took 1:59. The latter published 169 frames (49 source + 120 new), and Studio selected
the new five seconds with source-in 49/24 seconds. Both outputs included stereo 48 kHz audio and
decoded without errors. The motion take retained the scene but crouched more than requested.
These comparisons used the accepted H3 predecessor; they do not qualify LTX-to-LTX movie chaining.
Reference + endpoints + motion remains unsupported as one combined route. Final validation passed
271 Swift tests (two optional skips); core Python checks passed 333 tests, focused Studio/packaging
checks passed 117 tests, and cache/continuation checks passed 62 tests. Checkpoint parity and
physical low-memory hardware suites were not run.

Feature 2026-09-16: familiar native generation controls and LoRA stack parity. Native model,
render size and seed are directly accessible, and missing conditioning no longer hides valid
sampling controls. Native and Draw Things LoRA controls retain disabled strengths and expose
explicit Add/Replace group actions. Disabled entries are excluded from renderer requests.
Native H3 Turbo import records profile/layout/auxiliary-grid metadata and resolves a supported
four-evaluation schedule while retaining the user's standard step override for restoration.
Switching backends retains each native engine's selected model and sampling settings. An opt-in
headless H3 continuation contract saves bounded synchronized latent tails with verified provenance
and trims the repeated video/audio prefix. Studio now exposes native **Clip Continuity**: independent,
visible-frame matching, and experimental motion/audio continuation. H3 uses saved native context;
LTX 2.3/2.5 use their compatible source-tail extension adapters. Source take/trim changes invalidate
preparation and late results remain inactive versions. Export retries preserve accepted source
snapshots. Known text-only H3 and LTX model/task limitations are filtered before selection.
Automatic non-Custom model selection prefers matching tasks and compatible distilled LTX profiles.

Earlier continuity validation: combined core/Studio/workflows/remote checks passed 2,677 Python tests,
Studio and Draw Things Swift suites, lint, workflow and catalog checks. Installed-model preflight
passed for H3 four-step Turbo context saving and both frame/motion modes on LTX 2.3 and LTX 2.5.
Live Studio inspection verified context saving, successful preflight and preservation of the movie.
That initial increment did not include native generations; the subsequent exercise is recorded above.

Earlier controls validation: combined core/Studio/remote checks passed 2,285 Python tests and 292 Swift tests
(two optional skips), plus lint, workflow and catalog checks. Final continuation preflight changes
also passed focused regression checks. Release packaging and isolated live UI checks covered four-step
Turbo, restoration of standard Steps, retained strengths, group save/replacement, and controls for
incomplete first/last-frame inputs. Native Turbo render quality, continuity seams, actual performance
and memory fit were not measured in this increment.

Fix 2026-09-16: reimporting a reclassified Director subject now matches its stable workflow source
and item ID, including older kind-bearing keys and aliases retained by object reuse. Existing
project UUIDs, reviewed edits and references are preserved. Ambiguous legacy duplicates stop the
transaction with guidance to merge them first. Regression tests cover reclassification, legacy
keys, merged aliases, separate workflow sources and preservation on ambiguous import.

Fix and qualification 2026-09-16: constant-frame-rate movie export. Final assembly now conforms
the completed video stream to the project frame rate after concatenation and transitions. This
fixes AAC packet-padding/timestamp gaps that could leave a 30-second export with 718 frames
instead of 720 at 24 fps. Regression cases cover six AAC clips with cuts, dissolves, and the
rounded clip/transition durations entered through Studio's UI.

A local end-to-end movie exercise used Director's reviewed plan, the image workspace, seven
accepted endpoint images and six Draw Things H3 first/last-frame clips. The finished export was
verified at 30.000 seconds, 720 frames, 24 fps, 768×448, with stereo audio; full decoding passed.
Collect Media produced an editable project with all referenced media present. Planning, image
review, clip generation and assembly were operated through Studio's front end. This qualifies
that exercised local route, not automatic movie generation or every backend/model combination.

Feature 2026-09-15: Director first-pass correctness and focused review. New guided movies use three
explicit reviews: Brief, Subjects and Shots. A detailed eight-review option remains available;
existing jobs retain their saved approval contracts. Focused cards omit empty groups and collapse
technical notes and source evidence. Unsaved edits block approval/import, batch approval is explicit,
and missing-model setup is the primary next action. Execution details remain in the inspector.

Scoped shot corrections merge only the requested fields, retain before/after evidence and cannot
change host-owned IDs or timing. Successful repairs are consumed so later continuity refreshes
cannot replay a stale correction. Short authored shots are copied directly, avoiding unnecessary
summary calls. Longer summaries validate recognized exact speech, speaker attribution and duplicate
quotes against source and approved answers. Unassigned clarification dialogue stops explicitly.
Full approved character descriptions reach shot planning. Classification receives source evidence,
and relationship requests include legal target IDs; final subject kinds remain human-editable.

A frozen two-movie production evaluation now runs the real guided runner, records full local-model
turns and SQLite checkpoints, and checks exact dialogue, identity, timing and scoped corrections.
Its default validation makes no model calls and is included in the workflow validation profile.
The small text-only suite measures specific regressions; scripted structural approvals are not
human creative acceptance. See `examples/director-production-evaluation/README.md` for limits.

Validation: combined core/Studio/workflow checks passed 2,217 Python tests and 229 Swift tests
(two optional skips, no failures), plus lint, workflow and catalog checks. A separate release bundle
built and passed signature verification. Isolated live UI checks covered setup, focused subject and
shot review, draft guards, manual editing, batch approval, deterministic prompt compilation and
reviewed-plan import. The normal app executable, Info.plist and image workspace remained unchanged.

The final local Qwen run completed one of two initial plans; its completed plan passed the frozen
exact-text/identity/timing checks. The other stopped on extra model-generated timeline fields.
The attempted action correction stopped on changed quotation punctuation and retained saved output.
Whole buildings were still classified as sets. These measured failures remain open: stronger source
preservation does not yet establish reliable first-pass completion, successful correction, or a speed
gain. Separate action/dialogue contracts, more reliable response shapes and broader fresh movie
evaluation are the next priorities. No native video or low-memory qualification was run.

Feature 2026-09-15: Director foundation and independent assistant setup. Director is now a visible
toolbar entry; the Workflows menu and portable execution contracts remain. Intake, navigation and
unfinished typed reviews persist separately from approved output. Opening Director preserves the
movie's restore snapshot, including a never-saved first movie. Saved reference additions, removals
and ordering no longer require a model call. Imports validate their definition, effective inputs and
checkpoint identity before replacing the current session. Failed/cancelled execution retains saved
results, while document-session and input guards reject late results for changed targets.

Repeated interruption cannot make an unfinished coverage review ready. The local helper checks
exact text/image token budgets before loading weights, preserves source on overflow and reports
non-retryable context/model errors. Available preparation/load/prefill/decode metrics reach the
SQLite turn archive. Human review records preserve exact scoped changes, approval states and
revision/model evidence in the same atomic checkpoint; they grant no training consent. Scoped
history avoids repeating the entire inventory for every item approval.

Set up assistant can reuse a configured external checkpoint, locate installed files or download the
pinned 4.89 GB Qwen3.5 4B checkpoint independently of the Draw Things app. Transfers support resume,
cancellation, free-space checks, SHA-256 verification and existing-file protection. Explicit local
text/image health checks are separate from downloading and wait while a Studio job is active.

The new model-free evaluation harness includes 60 synthetic cases across ten task categories,
independent expected answers, source-group splits and versioned reports. Frozen corpus revision 2
with concise output templates passed 34/40 development and 15/20 holdout strict contracts on the
tested 256 GiB Mac. Production prompts differ; these are small-task diagnostic results, not an
app-wide accuracy or low-memory qualification. Remaining failures include exact text/formatting,
one taxonomy choice, one relationship ID and one route label.

A separate bounded Qwen3.5 MLX 4-bit experiment trained a language-only adapter, saved it and
reloaded an identical loss while preserving vision weights. Vision inference failed before training
in the installed MLX-VLM runtime, so alternate-runtime vision, task adapters and training remain
unavailable in the app. No dependency patch, embedding service or resident model service shipped.

Validation: combined core/Studio/workflow/remote profiles passed 2,449 Python tests; 25 targeted
follow-up tests cover subsequent fixture/documentation findings. Final Studio Swift reported 214
tests with two optional skips and no failures; helper Swift passed 41 tests. A separate
release bundle with corresponding helper source passed signature verification. Live isolated UI
checks verified external-model reuse, explicit text/image health, model selection and first-movie
draft recovery after full relaunch. The running app bundle was preserved. No native video render,
checkpoint parity, clean-Mac install or low-memory hardware qualification was performed.

Subsequent priorities: broaden production quality evaluation, measure model/cache reuse with shared
memory coordination, and add a consent-aware SQLite learning data layer. Full generation/timeline
integration and adapter activation require separate validation.

Product direction 2026-09-15: WeeTodd Studio is the primary standalone macOS app; the maintained
ComfyUI nodes are a secondary interface to the shared engines. Draw Things is central to fast
inference and model reuse. Native MLX engines supply additional models and advanced tasks beyond
the current Draw Things integration, with LTX 2.5 an existing example. Additional native models
should address useful feature gaps. Future Draw Things model discovery still requires verified
task/configuration mapping; compatible installed weights are reused only by supported native loaders.

The public project and repository name is WeeTodd Studio / wee-todd/WeeTodd-Studio. The README now
starts with standalone setup, backend choices, shared-model limits and product workflows, while
retaining the full node/native reference and workflow catalog. App, headless, authoring and agent
guidance follow the same direction. Package/import/node IDs, saved projects and existing checkout
paths remain compatible. This is a positioning/documentation change, not new inference support.

Validation: 1,827 Python tests passed across core and Studio; Swift reported 191 tests with two
optional skips and no failures. Markdown lint, 133 local documentation links, the generated node
catalog and portable workflow checks passed. The 33 changed ComfyUI workflow files contain only
installation-prose changes. A separate release build and signature verification passed; the running
app was preserved. GitHub repository naming/description and the local origin URL were updated;
source/documentation edits remain uncommitted. No generation or dependency installation was run.

Fix 2026-09-15: Project review reliability and editor responsiveness pass. Open/New now isolates
undo and document identity, preserves departing dirty work in per-session Recovery snapshots, and
updates the active startup snapshot immediately. Native and Draw Things clip preparation capture
the original document/session and inputs. Late results preserve edited clips as versions and report
saved artifacts when their destination was closed or deleted. Valid reviewed preparation is reused.
Render versions retain usable source intervals; known trims and extension offsets survive version
switches. Ambiguous legacy append offsets cannot be reconstructed and start at the selected known
usable interval. Movie transport uses the complete movie duration.

Attachment hashing now streams off the UI thread, deduplicates reads and caches by file revision;
Draw Things submission rechecks contents and explicit preparation retries transient I/O failures.
ImageIO previews decode to bounded sizes with a shared 64 MiB cache. Main editor controls have
accessible labels, current-shot empty-state guidance and a readable preparation summary.

H3 convenience generation now reuses staged text/sampler/direct AV publication and unloads on
success, failure and cancellation. Changed resident AdaLN schedules reload discarded transformer
weights. LTX sampler wrappers release weighted ownership before component swaps, including DFR;
unsupported chain modes reject before component inspection or prompt encoding.

Public/local guidance now reflects supported Draw Things and planning scope, narrower skill routing,
correct workflow cache semantics, expanded staged review coverage and safe worktree hook resolution.
Tracked validation profiles cover core, Studio, workflows and remote integration without duplicate
checks within an invocation. CI runs those profiles. The owned installed hook was backed up and
migrated through the tested installer. Packaging supports a separate output bundle.

Validation: 2,399 Python tests passed across the four profiles (1,745 core, 369 Studio/remote,
285 workflows); all six shipped Studio definitions validated. Studio Swift reported 191 tests with
two optional skips and no failures; Draw Things Swift passed 38 tests. Local hook/gate tests passed
10 cases. Compilation, broad source lint, Markdown lint, catalog checks and portable H3 API preflight
passed. A separate release bundle built and passed signature verification; the existing app executable
and Info.plist remained byte-identical. No staging or commit was performed.

No real-model renders, checkpoint parity, peak-memory benchmarks, clean-machine installation or
remote CI run were performed. Workflow draft/resume and AI-assistant usability/performance redesign
remain the next review; this pass changes their documentation/validation only.

Feature 2026-09-15: Execution history now includes a full Model turns inspector for each step.
New model requests and cached-response uses retain exact system/input text, returned text, image
references, runtime/settings, elapsed time, errors and recorded JSON/schema validation. Rejected,
cancelled and interrupted attempts remain inspectable; returned text is not treated as approval.
The separate local SQLite archive is capped at 16 MiB (or less under the job's storage budget),
reserves response capacity before generation and stops explicitly when full instead of rotating
away transcripts. Images and weights are not copied. The viewer pages metadata and loads one
selected transcript off the UI thread, with System, Input, Response, Images and Details tabs.
Older checkpoints expose retained full messages and label missing fields and chronology honestly.

Validation: 227 Python workflow/review/bridge/packaging tests passed; 160 Swift tests ran with two
optional skips and no failures. Coverage includes malformed-response retries, cancellation/cache
reuse, interrupted calls, archive limits and read-only SQLite inspection. Release packaging and
signature verification passed. The rebuilt app displayed all 22 retained description-review
responses from the user's current run; full system/input/response text was verified in the UI
without running generation or changing approvals.

Feature 2026-09-15: Workflow Execution history opens a read-only inspector during and after runs.
It explains registered operations, displays stored step durations and recursively counts retained
model responses, including object-level child records. New checkpoints keep bounded activity entries
for execution, explicit/upstream regeneration, changed-input execution, resumption and saved-step
reuse. Activities record real model requests, cached responses, outer step retries, call timings,
short task/system excerpts and errors. Manual description/coverage reviews have separate entries.
Failed and cancelled steps now retain elapsed time. History remains observational and does not
participate in approval or cache decisions.

The inspector reads checkpoints off the UI thread every two seconds and updates live elapsed time
once per second. Retention is capped at 40 activities and 20 details per activity with explicit
omission notices; no model/media copies or unbounded diagnostic files are introduced. Old checkpoints
show their actual saved timing/responses and explicitly lack reconstructed retry/regeneration history.
Returned model text is distinguished from validated step output; outer retries and per-object calls
are not conflated. Process interruptions disclose incomplete timing.

Validation: 222 Python workflow/review/bridge/packaging tests passed; 157 Swift tests ran with two
optional skips. Tests cover retry persistence, cancellation timing, zero-model-call resume, history
bounds, child-record calls, separate manual reviews and backward-compatible decoding. Release app
packaging and signature verification passed. The live UI loaded the user's existing subject-review
checkpoint without rerunning or editing it: extraction 14.0 s, linking 35.7 s, description review
101.6 s (22 retained responses), coverage 70.6 s. New telemetry was exercised with deterministic
backend fixtures; no new real-model render was required for this change.

Fix 2026-09-15: Draw Things Keychain credentials are reused in memory for the Studio session.
Concurrent successful reads share one lookup. Credential writes/removals invalidate cached access,
including failed mutations; denied and missing reads remain retryable. Clear Session Access in
connection settings forgets cached credentials without deleting saved keys. Saves/removals and
session clearing run off the UI executor as reads already did. No new credential file or relaxed
Keychain ACL is introduced.

Packaging accepts --signing-identity or WEETODD_STUDIO_SIGNING_IDENTITY, remembers successful
choices in ignored build settings, signs nested tools before the app, and verifies signatures.
An unavailable configured identity fails without replacing the previous app or falling back to
ad-hoc signing. Ad-hoc builds remain available with an explicit warning. Stable authorization
across updates still requires a suitable installed signing certificate; this development Mac had
zero valid signing identities, so the verified local build remains ad-hoc. Certificate-signed
upgrade authorization and notarization were not tested.

Validation: 154 Swift tests ran with two optional skips; 35 Python Studio bridge/packaging tests
passed. Regression coverage includes concurrent reads, credential changes/removal, failures,
missing entries, session clearing, signing order, saved identity precedence and failed signing.
Release app packaging/signature verification succeeded. Studio reopened and the new Clear Session
Access control was exercised successfully while the saved credential remained available. No live
cloud generation or real-Keychain permission lifecycle test was run for this change.

Fix 2026-09-15: resuming an approval workflow preserves completed documents across runtime-only
changes (including a rebuilt Draw Things helper and changed output-token limits). Previously the
helper's file timestamp participated in the execution key, causing reviewed subject extraction
to rerun and overwrite descriptions/approvals after an app rebuild. A separate content key checks
the saved step definition, resolved inputs, upstream outputs and reference-image contents.
Legacy checkpoints derive that key from their saved metadata before current runtime metadata is
applied. Completed-step execution provenance remains attached to its original runtime. Explicit
regeneration and changes to source content retain normal invalidation behavior.

Validation: 216 Python workflow/review/bridge/packaging tests passed; 151 Swift tests ran with two
optional skips. Regression cases cover edited and partially approved outputs, legacy migration,
runtime/token changes, and reference-content invalidation. An isolated copy of the user's legacy
workflow resumed against the current helper without model calls or lost descriptions. Six lost
object descriptions were recovered by original IDs from previously observed text and saved
image provenance; unverifiable historical approvals/reference assignments were not fabricated.

Feature 2026-09-15: Draw Things image generations display streamed, approximate live previews
and sampling steps in Studio. The helper uses the pinned dependency's lightweight latent
visualization without loading another VAE. Model/channel validation and throttling keep previews
best-effort; unsupported or absent previews leave generation and final-image delivery intact.
One temporary PNG is overwritten per job. Preview revisions refresh the viewport independently
of saved result paths, and previews never become assets or conditioning. Studio clears preview
state/files on success, failure and cancellation; only the active job's preview path is accepted.

Validation: 151 Studio tests (two optional skips), 38 Draw Things transport tests and 86 Python
adapter/client/image/bridge/packaging tests passed. A real local FLUX.2 Klein 9B KV image run at
1280×768, 8 steps showed live previews while sampling, then replaced them with the finished image.
Exactly one asset was added; the temporary preview was removed and the saved draft and timeline
were unchanged. Transport fixtures verify preview delivery before final publication; bridge
fixtures cover success/failure/cancellation cleanup and rejection of unrelated paths. Local
preview rendering is qualified; DT+ preview availability and video previews are not qualified.

Fix 2026-09-14: image regeneration now exposes explicit Random each generation / Fixed seed
modes. The reported duplicate images had identical fixed seeds, requests and file hashes;
the previous Randomize seed button selected a fixed number once. New image drafts default to
random mode (-1), while restored/imported fixed seeds are preserved. New fixed seed is explicitly
a one-time choice. Each generated asset still records the resolved seed for reproduction.

Validation: 149 Swift tests ran with two optional skips; 63 Python adapter/bridge/packaging tests
passed; release packaging passed. Two consecutive local Z Image Turbo generations at 1280×768,
8 steps, with identical prompt and non-seed settings produced different seeds, fingerprints and
image hashes. The saved draft retained -1 and the preview updated to the second image. The active
reference workspace was left in random mode. No cloud generation was used.

UI fix 2026-09-14: after a reported zoom-related unresponsive state, a process sample showed
the main thread idle in the event loop and no active generation. The old transformed image was
visually clipped without explicitly limiting hit testing. It is replaced by a scrollable viewport
with explicit bounds and non-interactive image content. The reference editor opens larger and
allows resizing; reference setup, existing references, the mood board and prompt can collapse.
Tools collects secondary actions. Config and connection dialogs present on the active image
editor instead of competing with the parent approval sheet. New results reset canvas zoom.

Validation: 147 Swift tests ran with two optional skips; 33 Python bridge/packaging tests passed.
Native testing reached 300% zoom, scrolled the image and opened the model/Tools menus without
losing responsiveness. Config and connection dialogs opened above the editor and dismissed cleanly.
The user's movie and saved image drafts/previews compared equal to their pre-update snapshots.
The process sample did not demonstrate a CPU deadlock; this qualifies the zoom interaction fix,
not a claim that every possible source of app unresponsiveness has been eliminated.

Fix 2026-09-14: the image-model picker now discovers its selected connection automatically on
opening and connection changes. Previously an unloaded catalog showed only the saved model as
a fallback; clearing that selection made the picker look empty. Loading, failed discovery,
empty catalogs, missing saved models and incompatible inputs now have distinct messages.
The image list is independent of selected model and inputs. Failed refreshes retain the last
successful catalog. Discovery uses a separate lightweight bridge, combines concurrent requests,
and rejects replies invalidated by connection edits. Restoring/importing drafts no longer runs
picker-change handlers that clear models or LoRAs; explicit edits retain compatibility checks.

Validation: 147 Swift tests ran with two optional skips; 33 Python bridge/packaging tests passed;
release packaging passed. Native testing loaded all nine local image models after restart without
Refresh, selected Klein, cleared the selection and verified all nine remained, reopened Krea with
its selected LoRA intact, then cleared/reselected the connection. The temporary LoRA was removed;
the editor remains on the user's local connection with no model selected. Failure retention,
concurrent discovery and stale replies were tested with injected responses; no cloud or image
generation was submitted for this fix.

Fix 2026-09-14: reference previews use distinct subject/reference-number labels and path-based
accessibility identifiers. Workflow thumbnail rows use file identity instead of array indices;
popup presentation captures an immutable file selection and resets preview state for that file.
This removes duplicate `00000000.png` / subject-only labels and prevents a removed row from
retargeting a neighboring preview. The Candidates menu also numbers results. Existing files,
asset names and reference bindings are preserved. Regression coverage includes duplicate
filenames, label changes and removing the first reference: 141 Swift tests ran with two optional
skips; 33 Python bridge/packaging tests passed.
Native checks opened both numbered subject references, switched to a numbered mood-board
reference, and verified distinct Candidate 1/2 menu entries. Two existing mood-board entries
referenced the same original file; they correctly display the same image and were left intact.

UI 2026-09-14: reference thumbnails now open a large original-image preview with Fit,
Actual Size, zoom (10–400%) and scrolling. Workflow approval, project subjects, reference
generation and mood-board thumbnails share the same accessible preview button. Loading an
unavailable file shows an actionable message; opening or closing a preview does not alter
reference bindings or approval. The original image is loaded only when its preview opens.
Validation: 138 Swift tests ran with two optional skips; 33 Python bridge/packaging tests passed.
Release packaging passed. Native testing opened both saved character candidates independently,
verified original dimensions, Actual Size, zoom and scrolling, and returned with Escape to the
same unapproved subject. No generation was needed for this UI change.

Feature 2026-09-14: object approval now offers **Create reference…**, opening the existing
Draw Things image workspace with editable character, portrait, prop, environment, set,
wardrobe and custom templates. Existing references can be explicitly loaded onto the canvas
or mood board when supported by the selected model. Krea Turbo and Klein starting settings
are optional; model, LoRAs, prompt and generation parameters remain editable. The picker
includes recognized Draw Things image families and registered custom checkpoints. Unknown
future architectures still require transport support. Qwen Edit Plus/2511 supports mood-board
inputs; base Qwen Image does not. Structured pose/ControlNet authoring is not part of this change.

Each candidate is saved in Project Assets with source-object/template provenance. **Use as
reference** attaches it without calling Qwen or approving it. Changed reference bindings invalidate
dependent approvals and mark older image analysis stale. Per-object drafts resume separately from
the ordinary image workspace; references attached to reused library objects remain movie-local.

Validation: 276 focused Python workflow/bridge/packaging tests passed; 138 Swift tests ran with
two optional tests skipped; all 37 Draw Things transport tests passed. Release helper packaging
and the signed Studio app build passed, along with Ruff, compile and README catalog checks.
Native validation completed a local Krea 2 Turbo eight-step 1280×768 character sheet, attached
it during workflow approval, reopened its saved draft, then generated a four-step Klein 9B
candidate using that sheet as a mood-board reference. Both outputs are saved and attached with
approval still pending. Source descriptions and all non-asset movie fields remained unchanged.
The models produced usable three-view layouts but changed appearance details, and ear tips were
cropped; these are review candidates, not qualified identity-perfect sheets. Qwen Edit and the
other template/model combinations have contract coverage but were not live-rendered in this pass.

Fix 2026-09-14: a saved character review failed twice because Qwen cited image zero with no
attached images. No-image design drafting now uses five plain visual fields with host-owned
proposal labels. Legacy-shaped image claims without attachments become unverified proposals;
invalid real source/image citations are never rebased. Reference-image reviews retain their
provenance checks. Initial guided discovery now follows cited-name extraction with a bounded
per-object source-phrase pass, retaining established visual traits and clarification answers.
Classification cards distinguish a source summary from a reviewed appearance proposal. A host
check also flags named inventory objects inserted into another object's appearance, even when
the model critic misses the conflict; explicit object-ID references remain allowed.

Validation: 275 focused Python workflow/bridge/packaging tests passed; 135 Swift tests ran with
two optional tests skipped. Release packaging, Ruff, compile checks and README catalog checks
passed. A real no-image review in the rebuilt app replaced the saved JSON failure with a concrete,
unapproved character design. Both character reviews completed. The named-object guard triggered
a second corgi draft that removed the stolen crystal from its body. Source evidence and object IDs
were preserved, and the autosaved movie compared equal to its pre-test snapshot. Qwen's critic can
still miss vague or inappropriate choices; these results qualify error recovery and useful draft
generation, not automatic design correctness. No new image-conditioned review or video render
was run for this fix.

Follow-up 2026-09-14: Studio now defaults new movie workflows to **Create a movie**. Friendly
creative inputs and consequential identity questions precede extraction. Human gates separate
classification, inventory/library reuse, detailed visual design, coverage, treatment, shots and
H3 prompt drafts. Original source passages and app-issued object/question IDs remain protected.
Saved extraction-only workflows expose a separate guided entry instead of changing their definitions.

Guided extraction uses compact cited-name selections, classification uses bounded local slots,
and treatment uses compact action arrays with host-owned cast records. Guided shot planning receives
approved character/prop/set definitions and creative preferences. Exact approved locations are
checked. Final structural review precedes deterministic H3 prompt compilation; dynamic inventory
links do not force future scenes or hidden held props into earlier shots. Import requires all steps
to complete and all required approvals to remain valid. Compiled subject IDs map to the approved
inventory, including library replacements, and exact prompt drafts are retained in shot direction.

Local Qwen remains fallible: this qualification required explicit review corrections to object
links, an edible canapé described as a replica, invented character hardware, and several story/action
details. Model coverage notes can also be inaccurate. These are human-review gates, not an autonomous
factual-quality guarantee. No frames, movies, image sheets, cloud jobs, adapter training or timeline
generation are enabled by the guided planning milestone.

Qualification: the isolated cat-woman/corgi fixture completed every gate with six reviewed objects
and six five-second shots (720 frames at 24 FPS). The real Python checkpoint passed native Swift
decoding/import with no duplicate objects; shot references and exact H3 prompt text were retained.
135 Swift tests ran with one optional vision test skipped; 266 workflow/bridge/packaging Python
tests and 333 node/runtime/README/workflow regressions passed. Workflow Ruff, compile checks,
example validation and README catalog checks passed. Release packaging passed. This qualifies the
reviewed 30-second path; long-form story quality and automatic frame/video generation were not tested.
Native UI checks verified friendly intake, answer/save/approval gating, the legacy workflow entry,
completed-job reopening, and the full-width H3 review with import available only after its gates.
The original indoor-pool review was restored; autosaved movie JSON matched its pre-test snapshot.

Follow-up 2026-09-13: reviewed subject inventory v1.2.0 adds a final whole-inventory coverage pass
after description enrichment. Validated draft links and exact phrase-to-ID anchors are retained
even when another proposed component is invalid. Approved rows and their dependency closure remain
unchanged; proposals are separate. Missing objects can become movie drafts through an explicit
button. Movie/global library candidates include pinned identity and definition metadata; exact
same-kind name/alias matches are supplied deterministically and semantic matches remain advisory.
Explicit reuse on import checks source freshness and target dependencies, then remaps references
transactionally. Review catalog refresh is separate from frozen execution inputs. No media copies.

Local Qwen3.5 4B qualification: a fresh three-object run completed in 31.49 seconds / six calls,
retaining Mara → jacket (wears), Mara → compass (holds), two indirect phrase anchors, one missing
helmet suggestion on Mara, and one exact jacket library match. The model still emitted rejected
components and inaccurate advisory notes; these remain visible and never authorize reuse or
approval. Native UI verified semantic link navigation, library preview/choice persistence, review
catalog refresh preserving execution inputs, and unapproved missing-object draft creation. The test
draft was undone and the autosaved movie compared equal to its pre-test JSON. Automated qualification: 124 Swift tests
(one optional skip), 227 root-run workflow/bridge/packaging tests; backend review also ran its
252-test regression selection. Ruff and whitespace checks passed. Release packaging passed.

Follow-up 2026-09-13: native description editors now decorate linked object names, aliases and IDs
with inline links and description tooltips. Navigation remains internal to Studio. Presentation does
not insert markup or change approval signatures; ambiguous names stay unlinked. Workflow drafts
remain editable, and approved descriptions remain selectable for link navigation.
Validation: 119 Swift tests ran (one optional skip), including native text-storage tooltip refresh,
internal navigation and plain-text preservation; 33 Python bridge/packaging tests passed. Release
build passed. Native UI confirmed inline appearance and navigation on the isolated Mara/jacket
fixture; the original pool review was reopened without editing its contents.

Follow-up 2026-09-13: production-library foundation is implemented in Studio. Characters,
environments, sets, legacy locations, props, clothing and outfits have stable IDs, tags, typed
relationships and distinct placements. Sets select an environment parent. A metadata-only SQLite
catalog publishes immutable graph packages; movies import pinned snapshots without media copies or
silent replacement of conflicting definitions. Environment publication includes its sets; set/shot
resolution excludes siblings. Movie changes and shot-only appearance notes preserve global identity.
Transitive definition changes invalidate dependent description/reference/shot approvals. Workflow
imports validate IDs and bounds before transactional mapping to project IDs; merges/deletes preserve
reference integrity. The native library supports search, publication, package import/export and
related-object navigation. Shot exports now include linked media metadata and resolved snapshots.

The reviewed builtin v1.1.0 inserts project.link_subjects@1 before description review. It preserves
identity/evidence, proposes known-ID links and flags missing reusable objects without inserting them.
Multiple placements can target the same prop. The workflow editor exposes relationship roles and
placement notes; human approval remains authoritative. Saved old builtin definitions remain pinned.
Initial native qualification verified package import, environment publication, tag search, separate
set/prop grouping, parent selection and two placements navigating to one object ID. The temporary
fixture import was undone and the autosaved movie compared equal to its pre-test JSON; its temporary
catalog row was removed. The empty SQLite catalog occupies 12 KiB. No media/model copies were made.
Validation: 116 Swift tests executed with one optional test skipped and no failures; 184 targeted
Python workflow/bridge/packaging tests passed; Ruff and whitespace checks passed. A local two-subject
Qwen3.5 smoke test first exposed reversed “jacket wears actor” semantics. The direction check now
retries once, then preserves the original description with advisory needs_attention. The corrected
run completed in 12.01 s / three calls, retaining “Mara wears jacket” and rejecting the inverse.
No descriptions or references were automatically approved. This is assisted graph authoring, not
proof of semantic correctness; human review remains necessary.
Automatic sheets, endpoint generation/timeline application, saved appearance presets and explicit
library version reconciliation remain next milestones.


Fix 2026-09-13: corrected the subject approval dead end after a user edits a description.
Human approval now accepts the exact saved text independently of the optional agent report,
matching project-level approval. Failed or older agent notes remain visible; no model call or
rewrite is required. The subject card offers Save & approve description and stops before approval
if saving fails. Empty names/descriptions, incomplete steps and stale checkpoint revisions remain
protected. Existing approvals still lock editing until explicitly unlocked.
Validation: 144 focused Python tests and 106 Swift tests passed (one optional Swift test skipped).
The rebuilt app passed an isolated native UI edit → Save & approve test, preserving the agent
report and locking the saved correction without a model call. The original checkpoint was unchanged.


Follow-up 2026-09-13: subject descriptions now have a bounded drafting/independent-critique
operation, a reviewed subject-inventory builtin, and an individual review action for existing
checkpoints. The workflow UI accepts up to eight reference images, sends them to local Qwen3.5
4B for both stages, displays image observations separately from proposed additions, and links
those images into Project Assets on import without media copies. Existing approvals remain locked;
agent reports remain advisory to explicit human approval (see the approval fix above). Source-labeled visual phrases must
match their cited text; unmatched paraphrases/additions are conservatively reclassified as proposals,
so a valid citation ID cannot certify invented details. Project approval remains an explicit human
decision with the imported report visible. Documentation explains this distinction.

Live qualification on isolated copies of the pool case and an existing warrior portrait confirmed
image transport and useful visual observations (weathered steel, rivets, dark leather, facial hair).
It also exposed false-positive critiques, contradictory details and malformed critic JSON from the
4B model. These remained needs_attention, never automatically approved. The two-round limit is
intentional; this is an initial assisted review, not a guarantee of visual or factual correctness.
Native UI qualification verified the image picker, remove/re-add thumbnail controls, retained image
bindings on job reload, and review execution from Studio. Automated validation: 143 focused Python
tests and 106 Swift tests passed, with one optional Swift test skipped; release bundle rebuilt.
No new image generation, downloads or cloud CU use occurred. Original user checkpoints were not
rewritten by these tests. At that milestone, storage remained portable JSON; the production-library follow-up below adds a separate global SQLite catalog.


Reconciled 2026-09-09 against the local source and saved acceptance evidence.

Follow-up 2026-09-13: subject workflow results now default to native grouped review, ordered as
Characters, Environments and Props, then by name. Names/descriptions can be saved and individually
approved; IDs and source evidence/suggestions are read-only. The shared review backend protects
these fields and approved subjects, including older checkpoints without per-item state. Explicit
approvals transfer to newly imported project subjects; existing project edits remain preserved.
The project subject browser uses the same grouping. JSON remains an optional inspection view.
At that milestone, storage used portable project/checkpoint files. The later global catalog does not migrate or replace those files.
Qualification: 127 focused Python tests and 105 Swift tests passed (one optional Swift test skipped),
and the signed release app was rebuilt. Live native UI checks on a separate copy of the user's
21-subject checkpoint verified grouping, retained drafts across selection, name/description save,
approval locks, approval persistence after reopening and unlock. IDs/evidence/suggestions, other
subjects and the original checkpoint remained unchanged. The original result is reopened for review.

Follow-up 2026-09-13: investigated the reference-image picker showing a beachball without a dialog.
macOS logs show its open/save service exiting on `SecCodeCopyGuestWithAttributes` error 100002;
the running Studio process predates the replaced app bundle. Packaging now checks for the running
destination app before building and again before replacement, preserving it with a quit/retry
message. Regression tests cover a running app and an app launched during packaging. After normal
quit/relaunch, the native file picker opened successfully during subject-review qualification.

Follow-up 2026-09-13: first project-planning milestone adds optional project-owned subjects and
shots, individual description/reference approvals, source preservation, workflow import, native Shot
List editing, reference-image links and JSON export. Reimports preserve user edits and stable IDs;
subject/reference/timing/continuous-ending changes invalidate affected approval signatures. Existing
projects decode without planning data; no timeline clips are modified. Source extraction uses local
Qwen in bounded sections, host-assigned IDs and host-copied numbered source evidence. The original
14,707-character script completed in 78.1 s / 12 calls, producing 21 draft subjects and correctly
preserving the dog's natural brown eye and sapphire optical eye. Semantic duplicates and some
classification choices still need review/merge; neither completeness nor identity is automatically
certified. The checkpoint is about 100 KB. No media/model copies or cloud generation were made.
Dedicated DT sheet workflows, automatic endpoint generation/visual checks and timeline application
remain subsequent milestones. Native UI access timed out, so visual interaction qualification is
still open. The live inventory also passed actual StudioCore import, repeated-import deduplication,
and project save/reopen checks with all subjects left as drafts. Independent review identified and
fixed reference-file changes leaving shots approved, and missing completeness warnings at the
per-section extraction cap. These now have focused regressions.

Follow-up 2026-09-12: reproduced the 14,707-character screenplay failure. Qwen accepted 3,500
input tokens (4,096 limit), returned 142 output tokens without truncation, but omitted the final
object brace. This was malformed model output, not an input-length rejection. Parsing now repairs
only one missing outer closer before normal strict validation; truncated responses and incomplete
strings/inner containers are rejected. The latest rejected response is retained for diagnostics.
Explicit sequential `[Shot N]` scripts are summarized shot by shot, with count/timing validation
before weighted work. The unchanged six-shot script reached story approval in 31.0 s through the
shared runner. Source text and the original failed run remain preserved; summaries still need
human review for visual identity and action fidelity.

Follow-up 2026-09-12: movie planning v2 now uses typed story actions and clip states instead
of splitting prose. It plans exactly one distinct action per clip; longer outlines are expanded
one story phase at a time in batches of at most four actions. Reference images are observed once.
The story and clip-state steps pause for explicit human approval. Studio has native story/clip
editors, approval/unlock controls, and instructed repair of individual clips. The shared CLI
supports the same revision-checked review mutations. Item caches reuse unaffected approved clips;
changed ending states revalidate subsequent continuous clips, while independent cuts can reuse
saved choices. Character IDs, frame layout, JSON types and approvals are checked before saving.
Endpoint descriptions are assembled from reviewed cast/location/states without another model
call. Structure validation is distinct from human review; neither cached nor fresh model prose
is presented as a guarantee of creative correctness. Existing v1 operations/jobs remain supported.
Checkpoints retain new-call prompts/responses and runtime fingerprints without copying weights,
media or unbounded histories. See the workflow authoring guide for bounds and review commands.

Qualification found and rejected two weak approaches: stretching a few broad beats across clips
repeated actions; asking Qwen4B for a large JSON outline produced malformed output. Phase-scoped,
small batches produced a complete 12-action outline in 19.7 s, and the 12-clip state plan in
77.7 s total. This is slower than the earlier 54.4 s prose planner and is not a matched sampling
speed claim. The gain is explicit intermediate control, stricter structure and targeted reuse.
Some generated states still describe motion and one endpoint incorrectly returned to a dungeon;
creative review and correction remain necessary. The instructed repair corrected clip 10 to a warm
cottage interior and revalidated clips 11–12 in 14.5 s / three model calls. Approved clips 1–9
were byte-identical and retained approval. A manual final-state edit used no model call; unchanged
resume added no executions. The resulting plan has 12 distinct last states, 1,440 frames, valid
structure and a ~66 KiB checkpoint. Fixed cast descriptions are excluded from premature-event
warnings, preventing stable clothing from being misreported as a future story event.
Validation: 456 selected Python tests pass, including 21 focused structured/review tests; Swift
ran 93 tests, 92 passed and one opt-in installed-model test skipped. Separate local Qwen runs
qualified story/clip planning, instructed repair and cached resume. Independent review found and
verified fixes for stale approvals on Run next, reverting edits, and repair completion reporting.
Ruff, compileall, portable H3 preflight, README node catalog and the release package pass.
Native app inspection repeatedly timed out, so interactive testing of the new editors remains
open; the existing user app session was not force-closed. No model downloads or video renders ran.

Follow-up 2026-09-12: fixed the dungeon movie-planning regression. The allocator no longer
copies the entire story into every clip. It partitions timed visual/action sections into clip-local
sentences, keeps camera framing attached to actions, excludes audio-only directions, and validates
story time coverage. Untimed stories distribute actions in order; sparse plans are flagged rather
than padded with invented events. Endpoint requests put the assigned action after the starting
frame context. Review now warns about repeated actions/endpoints, early fades, and possible future
story phrases; these are bounded heuristics, not a semantic-quality guarantee. The runner version
invalidates stale cached plans on the next execution. Original user run records remain untouched.
Qualification: eight focused allocation/review tests and 435 selected Python tests pass;
Swift has 89 passing tests and one opt-in skip. A matched run using the user's unchanged dungeon
story and a fresh end-to-end Studio run both restored chronological progression. Studio completed
in 54.4 s (original 67.3 s), with 12 distinct actions/end descriptions, 1,440 frames at 24 FPS,
and valid shared endpoints. The old output has only three distinct last descriptions and is now
flagged for repetition. Live UI verified completion and cached resume without additional calls;
the corrected test job is left open. This supersedes the earlier UI-timeout qualification limit.
The review remains heuristic; generated visual identity/motion and movie rendering were not tested.

Follow-up 2026-09-12: portable workflow contracts now have a shared sequential executor and
handlers for all eight registered operations. Studio exposes built-in/imported definitions,
editable typed inputs and image thumbnails, step results, Run next/remaining, pause/resume,
regeneration and executable local job export. Prompt Assistant links to the staged editor.
The same runner serves the bridge and `scripts/run_studio_workflow.py`, with typed output
validation, bounded retries/calls/time, atomic small JSON checkpoints, exclusive run locks and
dependency invalidation. Existing models/media stay referenced in place.

Movie planning provides exact project-frame allocation and endpoint descriptions, with continuous
clips reusing the preceding last frame. It does not generate images/video or mutate timelines.
Task-adapter metadata supports LoRA/QLoRA declarations; tensor loading, training and dataset
export remain unimplemented. Local 4B testing exposed malformed nested endpoint JSON, so endpoint
writing was split into smaller text tasks and bookkeeping moved into the host. A model-generated
review incorrectly flagged a valid one-sentence edit; reviews remain advisory. Reference evidence
is now supplied only to edits explicitly requesting images, preventing old image subjects from
repeatedly overriding requested changes. See `examples/studio-workflows/README.md`.
Validation: 502 selected Python tests passed, including 21 executor tests and 38 contract tests.
Swift ran 90 tests: 89 passed and the opt-in installed-model XCTest was skipped; separate real
local Qwen text/vision workflow runs were exercised. Ruff, compileall, portable H3 preflight,
README catalog, wheel-only imports/resources, managed-dependency hash dry-run and release app
packaging passed. Live native UI inspection still returns a computer-use timeout, so the new
workflow panel has build/core-test coverage but awaits an interactive smoke test. The running
user session was not force-closed. No new video renders or tensor-training qualification ran.
Fresh final-code runs: staged editing with one reference completed in 27.40 s and produced
“A red fox stands in a snowy forest lit by orange fire.” A two-clip/10-second movie plan
completed in 21.95 s with 240 frames at 24 FPS and an exact shared endpoint. The story test
still advanced to its ending too early despite explicit time windows; semantic pacing and
still-frame prose need human review and further qualification before automated movie generation.
The structural pass is not a story-quality endorsement. Test records are small external JSON files.

Follow-up 2026-09-12: Draw Things requests now accept seed `-1`, resolving it in the shared
adapter before estimation and retaining the actual seed through generation/provenance; saved
Studio drafts and headless jobs keep the sentinel. Invalid seeds produce a specific message.
Prompt assistance sends the latest instructions on every run. Live reproduction found partial
instruction following rather than a stale request; revised prompt rules prioritize requested
edits over source preservation, and unchanged/repeated results are now explicitly identified.
The saved Klein canvas/LoRA setup passed two real local preflights with distinct resolved seeds
and the same 1,795 CU estimate. Two consecutive 4B vision requests through StudioCore's request
builder produced distinct blue-moonlight and golden-sunlight revisions in 10.04 seconds total.
Compound editing remains imperfect. The rebuilt app awaits live UI requalification because
computer-use access timed out during the restart attempt; the running process was idle.

Follow-up 2026-09-12: experimental local Qwen3.5 prompt assistance is wired into Studio's clip
prompt editor and image workspace through the existing optional Draw Things helper. It reads
installed 4B i8x/9B i5x stores in place, previews editable text, and protects changed destinations
before applying. It uses no cloud credentials and releases the request-owned
runtime on completion or cancellation. The 4B path now includes selected images via DT's
vision encoder and multimodal decoder; the 9B path remains text-only pending separate validation.
The assistant shows image thumbnails, role labels and inclusion controls. An existing model was
located in an external DT store, so no new download is required for the live qualification.
Real 4B inference now passes single-image description, two-image endpoint comparison, eight-image
numbered-reference identification, text-only generation and cancellation checks. The eight-image
case selected the correct references (1, 4, 7), took 5.60 seconds and peaked at 4.12 GB physical
footprint on the M3 Ultra. These are smoke tests; the endpoint prose contained some incorrect
pose/setting inferences, so explicit review remains necessary and broad quality is not qualified.

Follow-up 2026-09-12: Studio has explicit timeline First/Last Frame slots, task-before-model
Draw Things selection, and an image workspace with one canvas and separate ordered mood-board
thumbnails. Canvas generation strength, per-reference controls, sampling settings and compatible
LoRAs/groups carry through shared remote requests and headless exports. Drafts persist by asset
store without copying media. Draw Things config import previews supported fields and omissions,
links to official presets, and allows an explicit installed-model replacement. Keychain reads now
wait off the UI thread rather than freezing Studio during OS authorization.

Local image acceptance covered Klein 9B KV canvas plus two references, reference-only input with
two compatible LoRAs, and Krea 2 Turbo I2I at 35% and 75% strength. Studio and the exported CLI
Klein canvas/reference job produced byte-identical PNGs. Cloud image qualification is still open:
the current live refresh waited on a macOS Keychain permission dialog. Existing Cloud video
acceptance is a separate result. Control images, masks and full advanced-config parity remain
unimplemented; unsupported imported fields are never silently presented as applied.
The live config-import, group-save and restart-recovery check also completed generation from
the recovered draft. Validation passed 77 Studio tests, 31 transport-helper tests and 687
selected Python tests, plus the release app build and portable workflow/catalog checks.

Follow-up 2026-09-11: matched LTX 2.5 Q8-paged T2V at 768×448, 121 frames and the
8+3 schedule measured 64.20 seconds on the repeated baseline versus 64.18 seconds with an
experimental one-block GPU-weight lookahead. Both complete video/audio MP4s were byte-identical;
MLX peak remained 8.06 GiB. The first baseline's 72.65 seconds included additional encoder and
compilation warming, so its apparent improvement is not credited to lookahead. First-frame
conditioning measured 70.37 versus 71.35 seconds, and two-subject MSR measured 149.36 versus
148.32 seconds, again with byte-identical paired movies. The rank-450 adapter stress test at
strength 0.3 measured 110.09 versus 107.72 seconds with the same movie hash; that single 2.2%
difference is not a repeat-qualified speed claim. This does not justify changing LTX
acceleration defaults or porting the candidate to LTX 2.3. Tests used an M3 Ultra with 256 GiB,
fresh renderer processes, and unpurged OS caches; they do not qualify a physical 36 GB machine.
Matched LTX 2.3 comparison is now complete: distilled 1.1 Q8, Q8 QAT Gemma, 768×448,
121 frames, 25 fps, eight evaluations, CFG 1/STG 0 and linear trailing Shift 5. Native DT
measured 103.0 s from a fresh process and 87.0 s warm, with its canvas cleared before each run.
WeeTodd's staged/streamed research route measured 90.2/90.3 s; a resident variant measured
89.3/87.7 s but raised process physical-footprint high-water from 21.3 to 128.8 GiB. It is not
promoted as the default. DT peaked at 31.6 GiB on that physical-footprint counter. These are
single pairs with OS caches intact; DT and MLX quantization/RNG differ, so equal seeds do not
establish cross-engine pixel or quality parity.

The validated route is now available as an explicit LTX 2.3 single-pass distilled 1.1 T2V
preset in Studio, headless setup and ComfyUI Generation Config. It selects the actual versioned
checkpoint, requires no Dev alias or upscaler, keeps staged/streamed memory defaults, and exposes
Steps and Shift with fixed audio/video guidance. Existing two-stage and conditioning routes
remain unchanged. The normal headless client completed the matched case in 91.1 s with the
exact same complete MP4 hash as the research route, eight progress callbacks and no retained
runtime. Process peak RSS was 19.0 GiB and MLX allocator peak was 30.1 GiB (different counters
from physical footprint). Tests cover checkpoint selection, task rejection, controls, generic
LoRA target validation, real sampler evaluation counts and staged cleanup on success/failure/
cancellation. Selected Python tests (658), Swift tests (60), packaging and workflow checks pass.
Full-size saved ComfyUI execution and physical 36 GB qualification remain open.

The same validation found and fixed a headless LTX 2.5 LoRA result-serialization error:
inspection paths are now JSON strings, so valid adapter preflight/generation can save `result.json`.
No weights or strengths change. First-frame and ordinary-LoRA recipe examples now document the
existing shared format. Studio managed runtimes need their source refreshed to receive the fix.

Follow-up 2026-09-10: Studio now displays native stage/evaluation progress, elapsed/last-output
time, and measured statistics for new render versions. LTX 2.5 guided setup includes dedicated
IC-LoRA control, Ingredients and MSR routes with compatible attachment controls. H3 head/FFN chunk
settings propagate into deferred blocks. An experimental 0–16 GB retained raw-page budget is shared
by Studio, headless recipes and the composable ComfyUI sampler; default 0 preserves no retention.
Cache budget, quantized/LoRA parity and success/failure/cancellation cleanup have focused tests.
Full-size cache timing and post-fix chunk performance on a physical 36 GB Mac remain unqualified.

Studio generation selection now separates engine, task and preset. Supported sampling controls and
LoRA strengths are visible; unsupported CFG/Shift or fixed schedules are explained rather than
silently ignored. Preparation and export share a content-based recipe resolver, and Generate runs
preflight automatically. Legacy clips preserve their Custom recipe behavior. H3 acceleration
preferences have per-clip overrides, and headless resident sampling now forwards the selected
policy while retaining staged unloading before decode. A matched 512×512, 124-frame, 19-evaluation H3 test on a 256 GiB M3 Ultra completed in
574.7 s with paged weights and normal working buffers versus 1065.8 s for the saved lower-memory
baseline; movies were byte-identical. MLX stage peak rose from 6.64 to 6.98 GiB. Full residency
reduced total time to 535.1 s but raised the stage peak to 32.32 GiB. Studio exposes both choices;
automatic defaults stay conservative pending broader qualification. This does not qualify either
option on a physical 36 GB Mac. See [measurement conditions](studio/README.md#matched-h3-execution-measurements).

Community report, 36 GB M3 Max (user-measured, source artifacts not independently inspected):
LTX 2.5 T2V at 768×448, 121 frames, 8+3 distilled, paged Q8 transformer/Gemma completed in
164.8 s (reported sampling 148.9 s), with reported complete process peak 9.20 GB. H3 first-frame
generation at 768×448 using the Q8 vision encoder completed in 1750 s (sampling 1628 s), with
reported process peak 7.72 GB, staged peak 6.24 GB and 8 ms AV drift. This establishes a reported
baseline for those recipes, not universal physical-36-GB qualification. The older BF16 comparison
used 672×384, so its 552 s timing is not a matched-resolution paging benchmark.

Follow-up 2026-09-10: an experimental native H3 DT-file adapter now reads original row-int8,
palette8, ezm7 and F16 storage without writing converted weights. Guided setup creates metadata
references for H3 T2V; the shared headless/composable-node path reuses the existing sampler and
staged lifecycle. Qwen language layers load sequentially with bounded embedding-row reads. Video
and audio decoders load from the original combined DT VAE. Image/reference/audio-input modes,
other DT model families, accelerators other than verified MPP projections and 36 GB hardware are
not qualified. The complete
DT-weight headless render completed at 512×512/124 frames/19 Euler evaluations in 924.5 seconds
with an 18.07 GiB process-footprint peak and 8.3 ms A/V duration drift. All weighted runtimes
unloaded. Sampled frames were coherent; broad quality and a saved ComfyUI DT-file render remain
open. Block preparation took 320.7 seconds, including 184.7 seconds decoding DT codecs. No DT app
performance parity is claimed.

The subsequent Metal weight-preparation pass reduced that same DT-weight render to 724.5 seconds
(21.6% less time) with an identical MP4 digest. Process footprint fell to 16.35 GiB; transformer
MLX peak rose from 4.94 to 5.31 GiB, with the overall 12.53 GiB MLX peak unchanged. Original weights
remain read-only, one block stays active, and all runtimes unloaded. All 534 mapped native tensors
and the exhaustive finite-half-scale/int8 rounding check matched the CPU decoder. Block preparation
fell from 320.7 to 134.4 seconds; direct packed-int8 projections and physical 36 GB qualification
remain open. Existing DT-file recipes use this preparation path automatically.

Verified Automatic MPP projections subsequently reduced the same render to 651.4 seconds
(10:51), with byte-identical video/audio and unchanged 16.35 GiB process footprint. All four
projection signatures passed the runtime gate. New normal-memory DT recipes select Automatic;
existing clips retain their settings and can opt in through the Projection control.

The next memory pass retains original F16 video decoder weights with FP32 activations, omits
the unused encoder, and evaluates each decoder block before building the next graph. DT transformer
reads use bounded read-only mappings with ordinary-read fallback; no mapped weight cache is kept.
These paths share the same Studio, headless and composable-node renderer. The matched complete
512×512/124-frame/19-evaluation M3 Ultra (256 GiB) render peaks at **9.48 GiB process footprint**
(42.0% below 16.35 GiB) and **6.58 GiB MLX allocation** (previously 12.53 GiB), with a byte-identical
video/audio MP4 and all weighted runtimes unloaded. Total time is effectively unchanged at
649.7 versus 651.4 seconds. Both runs used fresh renderer processes and fresh prompt caches;
OS file caches were not purged. Block preparation fell from 131.9 to 109.4 seconds, but compute
time increased in this desktop run. Do not claim an overall speed improvement from this pass.

The following DT preparation pass batches one cached-modulation block's GPU decoding and native
layout/dtype conversion, retaining eager fixed-weight and initial AdaLN loading. The same complete
recipe finished in **612.7 seconds (10:13)** versus 649.7 seconds, with a byte-identical video/audio
MP4, 950 batches, all weighted stages unloaded and no retained weight cache. Preparation fell from
109.4 to 91.7 seconds. Process footprint stayed effectively flat at **9.51 versus 9.48 GiB**;
transformer/video MLX peaks remained 5.31/6.58 GiB. Total elapsed time was 5.7% lower in this single
desktop comparison; unchanged block computation also ran faster (449.5 versus 463.6 seconds), so
the full elapsed gain is not attributable to batching alone. This does not extend hardware or
saved ComfyUI DT-file workflow qualification.

The subsequent native-layout pass fuses compatible DT int8 Q/K/V and gate/up decoding, ordering,
and BF16 conversion while preserving intermediate FP16 rounding. Other codecs, rounding modes,
fixed weights and initial modulation loading retain the existing path. All 50 real blocks matched
bitwise. Alternating preparation probes fell from 4.10/4.12 to 3.41/3.39 seconds, with temporary
MLX preparation peaks reduced from 1.72 to 1.08 GiB. The same complete M3 Ultra render finished
in **605.3 seconds (10:05)** versus 612.7 seconds, with a byte-identical video/audio MP4, 1,900
fused groups, all stages unloaded, no retained weight cache and no persistent converted weights.
Preparation fell **91.7 to 78.4 seconds**; unchanged compute took 454.0 versus 449.5 seconds.
The observed total improvement is 1.2%, with peak process footprint unchanged at **9.50 GiB**
and transformer/video MLX peaks unchanged at 5.31/6.58 GiB. Qualification limits above still apply.

The next DT-weight pass overlaps preparation of one next block with current-block computation.
It reuses the existing executor and exact native weight values, with a worker-owned Metal stream
and read-only SQLite handles. Normal-memory, resolved-MPP, single-block, cached-modulation runs
with no retained-page budget enable it; low-memory, standard MLX, selected-block windows and
retained-page caches keep sequential preparation. Failure, cancellation and close drain the worker;
source-change errors release prepared arrays even while the exception traceback remains alive.

The integrated matched M3 Ultra render completed in **546.8 seconds (9:07)** versus 605.3 seconds
(10:05), with a byte-identical video/audio MP4, 931 consumed lookahead blocks and all weighted
stages unloaded. Sampling/setup fell **550.5 to 490.8 seconds**. Process footprint was **9.48 GiB**
and overall MLX peak stayed **6.58 GiB**; transformer MLX peak rose **5.31 to 5.91 GiB**. The observed
elapsed improvement is **9.7%**; a same-workload prototype took 570.7 seconds, so desktop variance
remains material. That prototype had a single 13.56 GiB footprint sample at shutdown; the spike
did not recur in the integrated run. Neither new attention tiles nor FP16 attention inputs improved
the measured baseline and neither was promoted. DT sampler parity, a saved ComfyUI DT-file graph
and physical-36-GB qualification remain open. Six I2V/FFLF setup notes now correctly identify
their selected vision encoder and explain the optional Q8 vision-paged selector change.

## Shared renderer and ComfyUI

The shared Python/MLX backend runs through ComfyUI or `scripts/render_headless.py`.
The headless process blocks ComfyUI and node-catalog imports. It accepts versioned JSON
recipes and records resolved assets, effective conditioning, results, and runtime unloading.
The catalog contains 128 nodes and 46 UI workflows. Static validation establishes portable
contracts; it does not establish that every workflow has local models and selected input media.

| Engine | Implemented | Qualification limits |
| --- | --- | --- |
| H3 | T2V, endpoint/timed frames, multimodal Ref2VA, audio-driven Ref2VA, external extension, generic LoRAs, FastH3 and VDN variants | Native Ref2VA A2V and extension have real renders. Extension visual quality needs further qualification. Accelerator/task combinations are gated. A2V generates a new soundtrack. |
| H3 Fun ControlNet | Loader, preprocessing boundary, resident/paged execution, nodes and headless transport | Synthetic and checkpoint-header checks only. No real control render qualification. Checkpoint availability and applicable terms remain separate. |
| LTX 2.3 | T2V, keyframes, A2V, Ingredients, Union/Motion controls, generic LoRAs, Dev/distilled video extension | Conditioned renders and longer distilled extension have evidence. Generic LoRAs support resident and streamed paths; specialized control combinations remain separately gated. |
| LTX 2.5 | T2V, keyframes, A2V, Ingredients/MSR, IC controls, external extension, refinement/upscaling, LoRAs | Full-length MSR and short extension have evidence. Temporal DFR remains diagnostic. Every adapter/precision/task combination is not qualified. |

Ten saved H3/FastH3/VDN/LTX candidate pairs were rehashed on 2026-09-07: all headless MP4s
match their ComfyUI controls byte for byte. These short fixtures establish local adapter parity,
not publisher parity, all-task coverage, perceptual quality, or universal performance.
Newer conditioning integrations have their own evidence and do not inherit this certificate.

The local model library supports inventory, a persistent registry, shared asset references,
recipe import, supported LoRA normalization, and preflight. It does not implement universal
model detection, arbitrary downloads/conversions, DoRA/LyCORIS, or arbitrary missing-file discovery.
Explicit guided setup now discovers supported component layouts and handles pinned catalog downloads
and selected conversions outside graph execution.

## Standalone graphical interface

Draw Things integration is experimental. Studio includes saved gRPC/cloud connections, Keychain
credentials, image generation into the existing asset stores, a Draw Things clip type, CU preflight,
first-frame image inputs, and compatible server-resident LoRAs/groups. One shared adapter serves
Studio, v3 headless jobs, and ComfyUI. The transport preserves separate video/audio tensors and
refuses silent completion for audiovisual models. Headless jobs run sequentially, verify completed
artifact hashes, and require a deliberate new attempt after an uncertain remote submission or loss
of an already-generated artifact.

Synthetic gRPC image and audiovisual generation passed through Studio's native interface. Real
FFmpeg tests establish transport, timing, failure handling, and media publication. A CLI job with
Studio closed also completed image-to-first-frame video generation and movie assembly with an
existing clip, dissolve, title, and supplementary audio. These fixture-based checks do not establish
model quality or live account compatibility. Offline resume reused both remote artifacts and the
same final movie hash after the fixture server stopped. Direct Cloud uses
a read-only free-request/PAYG check and fresh CU policy; unknown allowance stays blocked. The DT+
App Bridge has no verified free-only billing contract and cannot generate in this implementation.
A real LTX 2.3 cloud response exposed a finalization timing assumption: all 121 video frames and
230,880 stereo samples arrived, but the complete causal audio was shorter than the 24 FPS video.
The helper and Python bridge now recognize and independently validate the exact LTX causal count.
Offline recovery produced a verified 768×448 MP4 with all 121 frames and the original 48 kHz audio;
no cloud resubmission was used. A matching gRPC fixture covers the receipt path. This was an earlier
recovery qualification; subsequent fresh-cloud results and remaining live-account limits are listed
in the [current qualification table](studio/README.md#headless-jobs-and-qualification).
Ordered H3 image references and supported image-model mood-board inputs are now available.
Movie/audio references, LTX IC-LoRA conditioning and local LoRA conversion/upload remain unsupported
through this remote integration; see the current Studio guide for supported model/task combinations.
The optional helper distribution includes dependency source and rebuild/replacement instructions.
See [Draw Things setup and qualification](studio/README.md#draw-things--experimental).

WeeTodd Studio now lives in `studio/` as a native Swift macOS application around the shared
renderer. It includes the requested editor layout, full-window prompt editor, native Light/Dark
appearance, clip-state colors and prioritized Actions, three collapsible asset stores, movie/still/
sequence import, multiple audio tracks, titles/transitions, versions, split/extension/bridge tools,
project save/recovery and Collect Media. Movie settings resolve at clip level during finishing.

Movie and clip headless-job export embeds generation and finishing plans. `WeeToddCLI` or
`render_headless.py --job` executes them sequentially with preflight, cancellation, integrity checks
and resumable render/finishing stages. Studio can install a private native Python/MLX runtime using
pinned, hash-verified dependencies. It preserves other environments and shared model files.

Studio also has a model-filtered linked LoRA library and reusable named groups. LTX 2.5 groups
accept both LTX 2.3 and LTX 2.5 members; H3 and LTX 2.3 stay separate. Applying a group snapshots its
members and strengths into the clip; editing a template cannot mutate existing projects or exported
jobs. The shared renderer retains authoritative adapter and task checks. GUI validation uses synthetic
header fixtures and establishes editing/transport behavior, not visual quality.

Guided setup now provides built-in H3/LTX presets, header-based existing-model reuse, validated
recipe creation, memory-policy advisories and explicit pinned downloads/preparation. Interrupted
downloads resume; SHA-256 and staged output publication protect existing models. Final media
preflight remains required for image/reference clips. Published Vayden releases provide the H3 Q8
vision encoder and complete LTX 2.5 distilled Q8 component set. The catalog pins each release and
every file checksum; source conversion remains optional and applicable model terms are retained.
H3 setup also offers the matching text/image or genuine reference transformer, Q8 video VAE and
official task support files (audio VAE, tokenizer, processor and task manifest). Task-specific downloads
are filtered, and explicit downloaded transformer provenance is checked during discovery and recipe
creation. Each model row has **Import…** for a linked local file/folder and **Download…** for a
compatible package, with size/contents/terms reviewed before transfer.

The app is an initial development build. FFmpeg/FFprobe and optional RIFE still need configuration;
retail signing/notarization and clean-Mac qualification remain release work. MetalFX interpolation is
experimental and requires explicit depth/motion/camera guides. See [Studio usage and limits](studio/README.md) for the exact implementation boundary.

## Experimental clip motion enhancement

Motion Fidelity (De-Roping) is implemented for H3 clips through Studio, v2 movie/clip headless jobs
and two ComfyUI nodes. It is off by default. Adaptive latent analysis or uniform holds expands video
and pitch-preserved conditioning audio; same-resolution partial H3 denoising then recovers source
frame timing and remuxes the original soundtrack. Sources are retained and enhancement settings do
not invalidate base generation. Completed enhancement stages can resume after hash verification.

A dense H3 896×512 boxing reference used 19 source evaluations without FastVideo/VDN. Uniform 2×
refinement recovered exactly 124 frames at 24 fps with audio/video duration drift below 1 ms.
Matching frames retain the subject/action, but changes are subtle and fast-glove blur remains.
This is not general motion-fidelity qualification. Plain H3 T2VA repair
recipes only; LTX, imported-movie UI support, long-clip windows and regional edits remain gated.
See [controls, budgets and limits](studio/README.md#motion-fidelity-de-roping--experimental-h3).

## H3 reference paging

Qwen paging v2 retains vision features and releases vision before sequential language layers.
The shared renderer accepts these pages for Ref2VA and FL2VA, with an experimental genuine Ref2VA
Q8-paged Comfy workflow and Studio/headless recipe preparation command. Existing text-only v1
pages remain T2VA-only. Tiny resident/paged tests cover actual image/video execution in FP32 and
BF16, repeated requests, cancellation/failure cleanup and packed Q8 storage preservation.
Transformer preparation from the genuine 13-shard Ref2VA source measured a 5.86GB complete-process
peak. A one-image 640×384 saved Comfy graph completed 19 dense evaluations and published
124 synchronized frames, with a 21.70GB full-process peak on an M3 Ultra/256 GiB host.
The headless output is byte-identical, with a 21.26GB complete-process peak and all runtimes
released. A Studio clip job exported and passed CLI preflight. The final Python suite excluding
optional algorithm search passed 1,407 tests with one skip; all 14 Swift tests passed and the
release application was rebuilt. Physical 36GB Mac and broad reference-quality qualification
remain open. Full BF16-versus-Q8 model parity was not run.

## Source checkpoints

- `e7ddb75`: guided Studio/CLI model setup and both published preconverted Q8 downloads; includes
  `f3168af` for native setup, model discovery, verified preparation and recipe creation.
- `e31a27c`: shared headless renderer and conditioned ComfyUI workflows.
- `4c56ea2`: native Studio editor, managed runtime installation and resumable movie/clip jobs.
- Publication cleanup: shipped Python preflight no longer depends on ignored agent files. App
  packaging verifies a fresh bundle before replacing the previous build. Project/job exports and
  macOS metadata are excluded from Git; user data and existing runtimes are preserved.

## Checkpoint validation and remaining work

H3 download completion (2026-09-09):

- The genuine Ref2VA Q8-extended transformer is published under Vayden at
  `9f339718e571f181b9a4c3043916ec0f2dc24f00`. All 13 native source hashes, 51 page hashes/headers,
  and 114 remote release files were verified before publication.
- The expanded catalog includes text/image and reference transformers, the Q8 video VAE, and
  task support packages. Source terms remain included. Native task/component filtering has focused
  coverage; downloaded provenance rejects conflicting transformer selection.
- 464 selected Python tests and 32 Swift tests passed. The existing text/image transformer, video
  VAE and both support packages passed installation checks; H3 text/image/reference recipes passed
  component preflight. The published Ref2VA package then passed installation, discovery and recipe
  preflight, including exclusion from the incompatible image preset. The release app was rebuilt
  with all catalog entries. These checks do not establish a new generation or physical 36 GB qualification.
- After the Mac was unlocked, native UI checks passed in Light and Dark modes: per-component
  Download expanded/scrolled to the correct package, reference setup excluded the text/image
  transformer, file/folder Import linked the selected paths, and LTX 2.5 clearly displayed its full
  component bundle. LTX 2.3 kept Import available with unsupported downloads disabled. The original
  System appearance was restored. No new download or generation was started during these UI checks.

Earlier model-setup qualification (2026-09-09, `e7ddb75`):

- 431 focused/required Python tests and 30 Swift tests passed; release app rebuilt with both catalogs.
- All 115 Qwen and 207 LTX release files passed remote size/checksum verification. Both packages
  installed through the shared setup service using verified existing weights and downloaded support files.
- The installed Qwen package was recognized as vision-capable. The installed LTX package exposed
  all five components and produced a validated recipe. Native setup was checked in Light/Dark modes,
  including the zero-recipes starting state, real model discovery and recipe creation.
- README/node/workflow checks passed. These setup checks did not include a new generation run,
  full publisher parity, or qualification on a physical 36 GB Mac.

Earlier checkpoint evidence (retain each figure's original scope):

- 1,304 tests passed and one skipped in the suite excluding optional algorithm-search tests.
- The focused node/runtime/headless/library/workflow review passed 468 tests.
- README catalog and portable H3 API preflight passed.
- Full publisher-checkpoint parity and fresh expensive model renders were not run in this review.
- The backend checkpoint is `e31a27c`; its validation figures above retain their original scope.
- Studio adds 10 Swift document tests and 13 Python bridge/job tests, including real media exports.
  At that checkpoint, the full Python suite passed 1,318 tests with one skipped.
- A new LTX 2.5 job completed generation, finishing, title assembly, and verified resume.
- A fresh app-managed native runtime produced byte-identical generated and assembled MP4s for that
  one-second fixture. Both used the existing shared model files.
- MetalFX spatial + RIFE finishing and explicit-guide MetalFX interpolation completed with audio
  and verified output timing/dimensions. No general interpolation-quality claim follows from these tests.
- Publication cleanup adds six packaging/preflight regressions. The full Python suite passed
  1,324 tests with one skipped (optional algorithm search excluded); all 10 Swift tests passed.
- Motion Fidelity adds shared H3 timing/refinement, Studio and headless integration, and two
  experimental ComfyUI nodes. The full suite passed 1,345 Python tests with one skipped; all 12
  Swift tests passed. Real H3 generation/refinement, packaged CLI movie assembly/resume, and the
  Studio Enhance/source-comparison action completed. Broad motion/identity quality remains open.
- Remaining release work: retail packaging, wider model-setup coverage, and qualification on clean,
  lower-memory Macs.
- Qualify additional adapter/task/precision combinations before promoting them in the interface.

Local research and detailed historical reports remain outside version control by project policy.
Historical reports retain their measured scope; this file is the portable current-status entry.
