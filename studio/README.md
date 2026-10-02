# WeeTodd Studio

WeeTodd Studio is the project's primary product: a standalone native macOS app for planning,
generating and editing AI movies and image assets on Apple Silicon. Draw Things is a central
integration for fast inference; the shared native MLX renderer adds models and advanced features
beyond the tasks exposed by that integration. The maintained ComfyUI nodes use the same shared
engines and remain available separately. Studio does not need ComfyUI.

This guide covers the application. Start with the [project overview](../README.md) for product
direction and backend choices. This is a source-build preview, not a notarized consumer release.

## Start here

Source builds include a Swift MLX LTX 2.5 worker. In Runtime Settings, **Use Swift MLX for
LTX 2.5** selects native distilled 8 + 3 step T2V, first-image, first/last-frame and
single-audio-driver A2V rendering with an optional first-frame image and compatible
standard LoRAs, including LTX 2.3 adapters. Continuous scenes of
two to six shots with identical effective components and sampling settings also use Swift.
Any scene shot may use one first-frame image. One consecutive interval of the same
audio source can drive every shot, including shots with an image. Studio renders the
group as one movie and accepts all member ranges together. Experimental native setup profiles
now expose MSR, Ingredients and Union Control through Swift. Experimental Swift
Motion Track, CrossView and combined Ingredients control contracts are also implemented;
direct installed-worker and saved ComfyUI graph checks are recorded below. A corrected combined
CrossView/Ingredients Studio lifecycle and native exported CLI route passed; broader control
quality and the other Studio control routes remain pending. Other specialized IC-LoRAs
and dedicated DFR editor controls still use the explicitly selected Python route.
An experimental direct Swift worker can execute spatial DFR from a version-8
request with an installed Pixel-Spatial x2 adapter, including first/last images.
Version-9 direct-worker requests add one or two learned temporal x2 rounds; an
installed-weight two-round clip completed with normal color at the reviewed seams.
Swift model setup now offers DFR spatial, one-round temporal and two-round temporal
presets. Select installed components in **Studio Settings → Model setup**; the scanner
recognizes the Pixel-Spatial detailing adapter and temporal latent upscaler from their
headers, and the worker preflights the linked profile without Python. Such a profile
supports T2V, I2V and FFLF. Ordinary LoRAs, A2V, extension and continuous scenes
are not supported with this route; the dedicated Studio DFR editor controls are not
yet wired to it. A Studio-style two-round profile passed installed-weight worker
preflight, but a Studio-triggered render is still unqualified.
The worker reuses installed paged
weights and the shared renderer. Unsupported controls fail native preflight; turn the option
off explicitly for advanced workflows still provided by Python.

The render job reports sampling steps and decode progress, then shows actual decoded frames
at a maximum 640-pixel preview size. These previews do not load a VAE during denoising.
Cancel releases weighted stages and removes unpublished output. Each completed native take
gets a distinct output directory; late results retain the existing project/clip revision checks.
Decoder geometry follows the prepared job in both preflight and execution; the earlier fixed
temporal cap that rejected 10-second clips was a Swift bug. Duration and resolution still
have to fit activation/workspace limits. The worker calculates each stage’s allowance from the
job geometry and the Mac’s memory, reserving space for other applications, weights and caches.
It no longer rejects larger jobs solely because they exceed a fixed 12 GiB allowance; requests
that exceed the hardware allowance or engine limit fail preflight with required/allowed bytes.
Native movie inspection uses AVFoundation. Supported LTX profile discovery and preparation
also run in Swift without a Python installation. Imported distilled recipes and already linked
assets can be prepared, rendered and accepted directly. Standard LoRA headers are inspected
without loading weights; compatible LTX 2.3 adapters retain their order and strength.
Final isolated-app checks completed T2V, first-image I2V and first/last-frame FFLF through
prepare, decoded previews, render, acceptance, project save and reopen with Python unavailable.
For each checked recipe, the raw movie matched its headless and recipe-backed ComfyUI Swift run
byte for byte. These checks qualify the tested recipes, not every duration or resolution.
A later 768 × 448, five-second installed-worker regression repeated I2V and FFLF preparation,
two live decoded previews, acceptance and reopening without Python inference. I2V's opening
frame matched its input visually; FFLF landed at both endpoints (31.78/30.84 dB PSNR against
center-cropped first/last source stills). The Swift workers took 78.75/82.60 seconds and reported
5.38/5.13 GB peak process footprint, excluding FFmpeg. A saved ComfyUI T2V API prompt also
completed through the packaged Swift worker in 72.28 seconds at a 5.38 GB process peak.
One additional 49-frame, 512 × 320 Swift A2V test at 24 fps completed in 31.82 worker seconds.
Its 98,000-sample stereo 48 kHz output WAV matched the expected trimmed/padded driver by
SHA-256, and sampled frames showed one coherent speaking person. This is a single short speech
take, not phoneme-level sync or longer-clip quality qualification. An installed Studio run of
this recipe worked with Python unavailable, delivered two decoded previews, accepted and
reopened the take, and produced an MP4 byte-identical to the direct worker with the expected
audio SHA-256. A second 49-frame A2V take combined the same source audio with a first-frame
image. The Swift worker finished in 40.27 seconds at 512 × 320 and 24 fps; the accepted Studio
take reopened with Python unavailable, showed two previews and retained the source-audio SHA-256.
This qualifies that short recipe only, not long audio-driven clips or general lip-sync accuracy.
A higher-resolution A2V run used a voice created in WeeTodd's Qwen3-TTS and one first-frame
image. The installed Swift worker rendered 169 frames at 1280 × 768 and 24 fps in 293.34 seconds,
with two decoded previews and a 13.67 GB peak process footprint excluding FFmpeg. The take
reopened with Python unavailable. The user reviewed this sample favorably; broader voice and
lip-sync quality remain to be tested.
A separate five-second
768 × 448 Swift take with an ordinary LTX 2.3 LoRA at strength 0.3 in both stages completed
with distinct coherent output; that one adapter result does not qualify every LTX 2.3 LoRA.
Swift now also prepares LTX 2.5 after-extension and motion continuation from a bounded,
hash-verified audiovisual tail of the accepted source's visible interval. The installed
worker produced a one-second 384 × 256 extension in 35.44 seconds with 24 video frames,
exactly one second of stereo audio and a 2.98 GB peak Swift process footprint, excluding
FFmpeg. Four reviewed frames retained the subject and waterfront across the join. The
distinct 49-frame motion mode completed in 44.52 worker seconds at a 3.68 GB
peak Swift process footprint. Both modes passed Studio prepare, preflight, render,
two-preview, acceptance and reopening with Python unavailable. Longer continuity
and seamless sound remain experimental. A separate two-shot scene with first-frame
images on both shots and a continuous Qwen3-TTS audio driver completed in Swift at
384 × 256, 24 fps and four seconds. The direct worker took 57.82 seconds at a 3.52 GB
process peak, excluding FFmpeg; its exact cut began at frame 48. A Studio run prepared,
rendered, accepted both shot ranges as one take, saved and reopened the project with
Python unavailable. The experimental Swift route also supports MSR still images and one
Ingredients sheet through dedicated setup profiles. A dedicated Union Control preset accepts
one preprocessed Canny, depth or pose movie. Studio streams a quarter-canvas RGB24 guide,
resamples presentation timestamps to the target cadence and freezes its checksum. A source
shorter than the editorial clip fails before inference; the last source frame can fill rounded
model-grid padding. Union requires final width and height divisible by 128. The compatible
LTX 2.3 rank-64 Union adapter is active only in stage one
of the distilled 8 + 3 schedule. Experimental Swift Motion Track accepts one preprocessed
colored-trajectory movie with its dedicated adapter. CrossView accepts ordered, explicitly
labeled warp/source movies and requires publication audio from the original source; it can
also use one described Ingredients sheet after those guides. These controls freeze RGB24
inputs and keep task adapters in stage one, followed by a clean stage two. Nondefault
per-reference attention and size overrides are rejected. Source-audio preparation streams native float32 PCM without an additional lossy encode.
Direct installed-worker Motion Track and CrossView tests at 512 × 256, 33 frames and 24 fps
completed in 41.92/31.60 seconds with two decoded previews and complete stage-release events.
Peak Swift-process footprints were 3.09/2.95 GB, excluding FFmpeg. CrossView publication
matched all 66,000 stereo 48 kHz source sample frames by PCM bytes. Its warp and source guides
were intentionally identical for this execution check. Saved recipe-backed CrossView and
combined CrossView/Ingredients ComfyUI graphs completed real inference as well. Their published
PCM independently matched the true source (528,000/1,936,000 bytes), and direct/Comfy CrossView
video, audio and muxed movies were all byte-identical. These are
execution checks, not approved camera/identity quality. A separate corrected combined recipe
using a fitted described sheet passed Studio preparation, two previews, acceptance and reopening
with Python unavailable at 512 × 256/121 frames. Render and acceptance took 86.45 seconds
at a 4.11 GB Swift-worker footprint, excluding FFmpeg. Its exported native CLI job reproduced
all three media files byte for byte and resumed with zero new/one reused generation. The same
corrected frozen recipe completed as a saved, uncached ComfyUI graph, again matching all three
media files. These individual executions do not constitute a matched performance benchmark. That
Studio run preceded durable event archival; its lost early UI-log events remain a historical gap.
A separate real combined-worker cancellation sent SIGTERM after stage one’s first transformer
block. It returned cancelled/exit 130 in 0.08 seconds after the signal and published no output;
this checks worker cancellation, without a direct allocator-zero measurement.
Other specialized combinations require their explicitly selected supported route.
Native H3/LTX worker renders and preflights now archive events on disk beside the job in
`NativeWorkerLogs`, with unique job IDs and filenames. The UI keeps its 30,000-character tail.
Archives survive success, failure and cancellation; worker results expose `workerEventsPath`
and `workerEventsTruncated`. A 64 MiB cap or disk-write failure marks the archive incomplete
and keeps draining worker output. Older bounded-display logs cannot recover discarded events.
Ordinary H3 and LTX 2.5 model setup now scans selected folders, links compatible
installed components and creates native profiles without Python. Swift also downloads pinned
LTX 2.5 packages, direct H3 transformers/support, Qwen pages, video VAE, folded audio VAE
and tokenizer packages. Other source conversions, asset/library tools and other engines remain separate work; this is not yet a
fully Python-free Studio release.
The experimental Swift H3 worker has completed text-to-audiovisual, reference-conditioned
and independent audio-driven takes.
Direct worker tests now also cover a Swift-only version-2 latent continuation
pair and a Ref2VA external movie extension with a final-frame seam anchor.
Studio's **Match previous frame** now freezes the accepted source's last visible frame
for Swift H3. **Save motion context** and **Continue scene** use native version-2
context for text-only H3 clips. Installed-app frame and motion jobs passed live previews,
acceptance, saving and reopening with Python unavailable. The motion output matched its
headless and saved ComfyUI graph byte for byte. This proves route/lifecycle parity for
one recipe; joins remain visible. With Swift H3 enabled, **Extend → After** now prepares an
experimental 4–15-second Ref2VA extension. Choose the reference-capable profile and write the
continuation action; preparation preserves a complete structured continuation prompt or maps
the action, soundscape and music into its required sections. Studio freezes only the source's
visible ending, up to 175 aligned frames at 24 fps and a 256-pixel reference edge. The worker
anchors the new shot to its final frame and generates a new soundtrack. This route publishes
only new frames, starting at source offset zero; the source take remains unchanged. Extra media,
saved motion context and before-extension are rejected. Extension quality and matched speed
and memory remain under qualification.
The off-by-default Swift preparation path validates clip inputs, ordered references and profile
settings, then requires worker preflight before a take can render. Enable **Use Swift MLX for H3
(experimental)** in Runtime Settings to try text-to-AV, timed keyframes, or ordered Ref2VA with
up to nine still images, three movies and three audio-bearing references. A movie
contributes its first 7.3 seconds, resampled to 24 fps and trimmed to the VAE clip grid;
an embedded soundtrack from that interval contributes sound-reference latents.
Standalone audio used as an ordinary Ref2VA reference requires a visual reference. The
separate **Audio to video** task accepts one 2.5–15-second driver interval covering the
clip and optionally one opening image. It generates a new soundtrack and motion from the
source; it does not copy the original audio into the finished movie.
The H3 audio encoder has a short numerical parity test, but full audio/soundtrack output quality,
adapter stacks and production quality remain unqualified.
One mixed still/movie 768 × 448 test completed 73 frames with stereo audio in 739.99 seconds
at an 8.10 GB peak Swift process footprint (FFmpeg excluded). Appearance was coherent,
but its reference head turn did not reliably transfer. Multiple-movie quality and matched
performance have not been established. The Qwen visual-token and H3 packed-row limits
can reject a combination even within the per-modality reference counts.
One image plus 6.8-second Qwen TTS voice-reference take produced 73 frames and stereo audio
at 768 × 448 in 822.33 seconds with a 4.29 GB peak Swift process footprint (FFmpeg excluded).
The audio-reference encoder took 1.68 seconds; sampling took 595.46 seconds. The sampled
frames held a stable subject, while voice likeness and intelligibility still need listening
review. The earlier movie-reference test used different conditioning, so these are not
isolated speed or memory comparisons.
An image plus audio-bearing 73-frame source movie also completed a 73-frame, 768 × 448
Swift take with stereo audio in 841.25 seconds at an 8.10 GB peak worker footprint
(external FFmpeg excluded). Sampled frames held a stable subject with one speaking pose;
voice likeness, intelligibility and lip motion still require playback review.
An additional recipe-backed ComfyUI T2VA run used the installed FL2VA checkpoint without
image references and the four-evaluation Turbo LoRA. Its 768 × 448, five-second robot shot
completed in 579.94 seconds at a 5.23 GB peak Swift process footprint. The visual arm motion
is coherent, while generated stereo audio is subdued (-57.5 dBFS mean, -28.1 dBFS peak).
H3 FL2VA also prepares up to eight images at distinct frame positions, including interior
keyframes; three-image and eight-position installed-checkpoint preflights passed. If their
full-canvas Qwen patches exceed its token window, the worker scales only Qwen's visual copies
while retaining the full-canvas VAE conditioning images. One three-keyframe, 768 × 448,
73-frame, four-evaluation turn completed in 399.77 seconds at a 3.98 GB peak Swift process
footprint, excluding FFmpeg. Front, interior and final outputs followed the corresponding
images at 0.9902/0.9976/0.9956 correlation; broader timed-anchor quality remains unqualified.
H3 aligned this requested 2.5-second turn to a 73-frame, 3.05-second muxed movie. Studio
places the last reference on the final frame visible in the requested clip interval;
explicit headless `"last"` continues to mean the raw generated endpoint.
First/last-frame preparation and worker preflight have structural coverage. One
corrected real 768 × 448, five-second first/last recipe (124 frames, four evaluations, seed
20260927) showed a coherent front-to-profile turn; first/last endpoint Pearson correlations
were 0.9903/0.9957. The worker took 511.623 seconds under a concurrent app build, peaked at
4,720,223,400 bytes MLX allocation and 5,742,871,896 bytes process footprint, and used no
swap. Its stereo audio was nearly silent under a quiet-room prompt. This qualifies one
experimental visual recipe, not audible AV quality, broad FL2VA quality or speed parity.
The signed app's bundled worker produced a byte-identical MP4 on the same recipe. It took
552.882 seconds, including 414.207 seconds of sampling and 119.716 seconds of video decoding,
at the same 4,720,223,400-byte peak MLX allocation and zero swap. This pair is not a matched
speed comparison. A separate Studio lifecycle run of the same five-second recipe passed ordered
endpoint preparation, rendering, seven decoded previews, take acceptance and project reopening
with Python unavailable. Render and acceptance took 591.83 seconds; the resulting 768 × 448
movie decoded fully with synchronized video and audio. Its front, middle and profile frames
retained the intended turn. The worker reported 5.02 GB peak MLX allocation and 5.72 GB peak
process footprint, excluding FFmpeg. This proves this recipe's app path, not production speed or general
audio quality.
An otherwise matched prompt requesting chainmail rattles and breathing kept the visual turn
coherent but emitted audio at about -65 dBFS mean and -47 dBFS peak. A separately amplified
listening copy is available for review; the generated soundtrack itself remains too quiet
for normal playback, so FL2VA sound quality is not qualified.
An installed two-reference Studio take passed preparation, worker render, seven decoded previews,
clip-source adoption, project save and reopen with Python unavailable. Ordered reference preflights
passed for every count from one through nine, and a real nine-reference exported Studio job
completed. Those one-, two- and nine-image visual observations are historical: the old
still-reference conversion inverted image rows. They verify execution and Studio lifecycle,
not likeness. The H3 route remains experimental; broader performance qualification and
media-reference tasks remain open.
A separate one-reference packaged-worker take completed with synchronized video/audio, decoded
previews, and no swap; its Studio UI lifecycle has not yet been exercised end to end. After
the orientation fix, the same-seed one-image Beowulf recipe completed again at five seconds,
768 × 448, 124 frames and four evaluations. Its fully decoded frames showed coherent boxing,
darker hair and beard closer to the portrait, a clearer face during the jab and glove contact
with the red bag; white shirt stripes were invented. The worker took 750.707 seconds under a
concurrent app build, peaked at 4,889,500,852 bytes MLX and 5,731,812,312 bytes process
footprint, with zero swap. Its non-silent stereo audio measured -29.3 dBFS mean and -2.6 dBFS
peak. This is one corrected recipe, not broad likeness or speed qualification.
The corrected packaged worker then admitted every count from one to nine ordered stills and
rendered a real nine-image, 768 × 448, five-second Beowulf boxing clip. It took 882.62 seconds,
peaked at 6.06 GB Swift process footprint, and retained a recognizable subject in the reviewed
frames. The one-image run took 750.71 seconds at 5.73 GB under a different workload, so this
pair does not establish an isolated scaling benchmark.
Swift H3 preparation accepts up to four compatible LoRAs in a stack. One installed
FL2VA test combined the full-rank and resized Lightx four-step adapters at strength
0.5 each. The signed worker passed preflight and generated a 73-frame, 768 × 448
audiovisual robot clip in 403.78 seconds, with a 4.10 GB peak Swift-process footprint
(external FFmpeg excluded). Three inspected frames show a coherent arm raise. This
qualifies that pair's execution only; neither arbitrary LoRA combinations nor a
matched speed improvement has been established.
Still-image asset import uses ImageIO metadata inspection without decoding full-resolution pixels
or invoking Python. MOV/MP4/M4V movies and MP3/M4A/WAV/AIFF/CAF audio now load track metadata
through AVFoundation, so these H3 reference assets can also be linked when Python is unavailable.
Other formats, text and LoRA files still use the existing inspection bridge. Importing a linked
asset does not establish that its codec or duration is valid for a specific generation task;
the selected worker checks that during preparation.
To try the experimental Ref2VA route, choose **MiniMax H3 → Reference video**, import up to nine
images, three movies and three audio files into the clip asset store, then choose
**Use in clip → Appearance** for images/movies and **Audio · reference sound / voice** for
audio. Movie references use only their first 7.3 seconds and include any soundtrack in that interval;
audio references use at most 15 seconds and need an image or movie alongside them.
Write a complete H3 prompt, select the installed compatible Ref2VA profile, and use
**Prepare clip** to inspect the exact settings and run preflight before generating.

| Your goal | Go to |
| --- | --- |
| Build and launch the standalone app | [Build and open](#build-and-open) |
| Generate through a local Draw Things server or Cloud API | [Draw Things setup](#draw-things--experimental) |
| Reuse supported installed Draw Things weights with native H3 | [Local H3 model reuse](#reuse-local-draw-things-h3-models) |
| Set up native LTX 2.5, LTX 2.3 or H3 | [Guided model setup](#guided-model-setup) |
| Choose engine, task and sampling controls | [Clip generation](#clip-generation-controls) |
| Continue motion across clips or render an LTX 2.5 scene | [Clip continuity](#clip-continuity-in-studio) |
| Generate image assets or import Draw Things settings | [Image workspace](#images-clips-and-loras) |
| Generate sampled-reference speech and mix voice/music | [Voice and audio mixing](#native-voice-and-audio-mixing) |
| Compose music and drive video from a song | [Native YuE2 music](#native-yue2-music) |
| Plan a movie with reusable subjects and references | [Guided movie planning](#guided-movie-planning) |
| Use local Qwen models already installed by Draw Things | [Prompt Assistant](#local-qwen35-prompt-assistant) |
| Automate a movie or use ComfyUI graphs | [Headless jobs](#headless-movie-and-clip-jobs) / [Nodes](../README.md#comfyui-nodes) |

## Inference and model storage

Choose **Draw Things** in the **Generation** menu to submit to a configured Draw Things endpoint.
Choose **WeeTodd (local)**, then **H3**, **LTX 2.3** or **LTX 2.5** in the **Model** menu for native
execution. Compatible installed components are selected automatically, with no template step. These are
explicit choices; Studio does not silently substitute one backend for another.

Use the Draw Things route first when it supplies the needed model/task. Native execution is useful
for features such as audio-driven video, video/reference conditioning and model-specific controls
where implemented. LTX 2.5 is already a native engine; the current Draw Things adapter covers
LTX 2/2.3 and selected H3 video tasks. Future Draw Things models should be accessible through the
existing UI when their capability and configuration mappings are compatible. Advanced native
features remain useful even after the same model becomes available through Draw Things.

**Sharing weights is a separate choice from selecting the inference engine.** Supported H3 model
files in a Draw Things installation can feed the native renderer without copying/converting the
whole checkpoint. That initial direct-weight path is text-to-video with generated audio; it does
not yet qualify all native conditioning tasks. The local assistant separately reuses supported
Qwen3.5 files. Other models require compatible native components. Keep model files in a stable,
readable location, and let setup validate the particular component layout and task.

## Native voice and audio mixing

Open **Movie → Generate Voice…**. Speech inference runs inside WeeTodd's own MLX engines through
its local runtime, with no speech API or external inference server. Configure shared model locations
under **Runtime Settings → Speech models**, using **Add installed model…** or **Download speech
model…**. In Voice, choose **Family** (Fish or Qwen), then an **Installed model** variant. Each
family remembers its last selection across movies and app launches. Model paths belong to Runtime
settings; new voice drafts store a model ID, while completed takes retain their resolved request.
Older draft model locations migrate into Runtime settings when opened. Downloads are pinned and
verified; unsupported tensor layouts fail inspection before model allocation. Precision follows
the installed checkpoint. Removing an entry from Runtime settings does not delete its model files.

| Engine | Supported local weights | Reference modes | Master audio |
| --- | --- | --- | --- |
| Fish S2 Pro | MLX community S2 Pro 8-bit or BF16 | Audio + transcript; synthetic voice | 44.1 kHz mono |
| Qwen3-TTS Base | MLX community 12Hz Base 1.7B or 0.6B, 8-bit | Audio + transcript; speaker identity only | 24 kHz mono |
| Qwen3-TTS CustomVoice | MLX community 12Hz CustomVoice 1.7B, 8-bit | Nine built-in voices; delivery instructions | 24 kHz mono |

Fish weights use the [Fish Audio Research License](https://huggingface.co/fishaudio/s2-pro/blob/main/LICENSE.md);
commercial use requires separate permission from Fish Audio. Qwen3-TTS weights use
[Apache 2.0](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).
These terms are separate from WeeTodd's source license. Original Fish `codec.pth`, arbitrary MLX
conversions, Qwen VoiceDesign, CustomVoice 0.6B, fine-tuning and speech LoRAs are not supported here.

**Sampled-reference workflow:** choose an audio/video sample, an existing audio asset, or the
selected clip's soundtrack. Enter the sample start and length (up to 60 seconds), choose its channel,
and audition the range. In Audio + transcript mode, enter the exact words spoken in that range.
Changing the range does not rewrite the transcript. Save a reference preset to reuse it in this
project. Qwen's speaker-identity-only mode accepts audio without a transcript, with potentially
weaker matching. Fish's synthetic mode needs no sample.

**Fish voice direction:** use **Pitch**, **Pace**, **Timbre**, **Accent** and **Voice description**
to guide a voice with or without reference audio. Describe vocal character, age or other qualities
in your own words. **Applied Fish instructions** shows the free-form tags sent before the script;
your script and reference transcript remain unchanged. These are model instructions, not fixed
acoustic measurements, and adherence varies. Set each character's direction under **Speakers**;
a line can override it. Clearing all fields leaves the model unguided. Switching to Qwen retains
the stored Fish settings without sending them to Qwen.

**Generation settings** has visible labels for Seed, Temperature, Top P, Top K and Maximum audio
tokens; hover over a field for its range and meaning. The dice button varies the seed. A seed does
not lock speaker identity across scripts. To reuse a generated voice, select its take under
**Audio + transcript → Use existing audio**, enter the words in the selected sample range, and
save a reference preset. Qwen CustomVoice offers its own preset speakers and delivery instructions;
Qwen VoiceDesign remains a separate, unsupported checkpoint in this release.

**Fish delivery tags:** use **Add tag** or the quick buttons above the script to insert square-bracket
instructions at the cursor. Emotion, delivery, timing and reactions are grouped; **Custom delivery**
accepts a short description. Tags remain editable in the script. **Auto Tag** uses the configured local
Qwen3.5 assistant to suggest sparse insertions for the current script/line, with a preview before Apply.
It preserves the original words, rejects invalid locations, and never analyzes the reference recording.
Assistant setup is shared with Prompt Assistant; no hosted Fish API is used.

**Qwen emotion:** the supported Base checkpoints do not accept Fish tags or natural-language style
instructions. Use an expressive audio sample plus its matching transcript to condition delivery;
identity-only mode is less suitable for transferring a performance. For direct text instructions, install
**Qwen3-TTS CustomVoice · 1.7B · 8-bit** in Runtime settings. Select one of its nine built-in voices,
choose an **Emotion preset**, or write a delivery instruction (up to 2,000 characters). Instructions
remain separate from the spoken script. CustomVoice does not accept sampled voices; switching back to
Base preserves your references. VoiceDesign and 0.6B CustomVoice are not supported.
The distinctions follow the [official Qwen model capability table](https://github.com/QwenLM/Qwen3-TTS#released-models-description-and-download).

**Conversations:** switch **Single voice → Conversation**, open **Speakers**, and name each character
and assign their reference sample. Add ordered lines, choose their speakers and set pauses (0–10 seconds).
A line can override its speaker's sample with another performance by the same character. Changing the
assigned speaker clears that line's old override. Fish tags and Auto Tag work on the selected line.
Qwen CustomVoice conversations instead assign a preset voice and delivery instruction to each speaker;
a line can override that instruction. Both families render each line with its assigned voice, then
assemble one dialogue take; speaker names are never spoken. This is sequential orchestration, not simultaneous overlapping speech or a
single native multi-speaker sampling pass. Up to 64 lines are supported. Individual line takes, seeds,
references, and sample-exact timing remain in the take's artifacts. Use **Add to clip** to place the
dialogue on the voice track alongside music. Switching modes preserves both scripts.

Enter the new script, choose a seed and optional sampling controls, then generate. Each take retains
its full native-rate master, generated codes, request, model/reference digests and completion receipt.
Stages release weights before the next component loads; success, cancellation and failure release
the active model. Completed takes never automatically replace or place earlier media. A token-budget
ending is labelled; audition it before deciding to use it. Speaking length is model-selected, not
an exact-duration promise. **Add to clip** creates a clip-anchored Voice region and trims its visible
length to fit; the complete take remains available. Moving/splitting its clip moves/splits that region.
Music stays independently editable. Collect Media includes voice references and take artifacts;
model weights remain in their shared library. Collected projects rebuild audio drivers from the new paths.

Select an audio track to set its role, gain, pan/balance, mute or solo. Select a region for its source
trim, length, volume and separate fade-in/out. Equal-power crossfades require overlapping regions
on the same track; use **Crossfade with next region**. Music tracks can enable voice ducking
(up to 12 dB, 20 ms attack, 250 ms release, -36 dBFS threshold). Mono uses equal-power pan; stereo
uses balance with unchanged center levels. Clip source audio has separate volume and pan.

Select an audio track and enable **Reverb** in its inspector. Choose **Room**, **Chamber**,
**Hall** or **Plate**, then adjust **Amount** and **Decay** (0.2–6 seconds). **Advanced reverb**
adds a dark/bright tone control and 0–100 ms pre-delay. Presets keep your amount; switching
reverb off preserves settings and restores dry audio exactly. Older projects open with reverb off.
The stereo effect covers the whole track, so tails continue across regions, splits and gaps.
Music ducking also lowers the reverb tail while voice is present. Preview, export and selected
audio drivers share the same effect. Tails stop at the movie boundary: leave space after the last
sound on the timeline to hear its full decay. Reverb does not lengthen video clips.

New projects share one 48 kHz stereo mixer for preview, export and driver preparation, with explicit
gains and a 0.95 peak limiter without automatic makeup gain. Solo is a preview audition control.
Older projects retain their legacy limiter makeup, symmetric fade cap and export-solo policy.
Mix edits coalesce for 150 ms; the current player item stays active while a replacement is prepared.
Decoded PCM has a 1 GiB disposable cache and preview mixes a 2 GiB eviction target, with active-file
leases and a short grace period; saved driver/export artifacts remain durable. Preparation uses
bounded memory and supports timelines up to one hour. Long timelines can take longer to rebuild.

For an independent native H3/LTX clip, open **Timeline audio driver** in its inspector. Choose
**Voice**, **Music**, or **Voice + Music**, select contributing tracks and prepare/audition the exact
mix. Gains, pan, fades, reverb, ducking and mutes are included; solo does not change the driver. The
Swift LTX 2.5 and experimental Swift H3 routes accept the prepared mix for an
independent A2V clip. H3 requires a compatible Ref2VA checkpoint and can also use one
opening-frame image. **Use timeline soundtrack (mute clip audio)** prevents doubled audio
while retaining the original generated movie. Preflight rejects missing, stale or silent
drivers. Native frame geometry may require a slightly longer reference than the visible
clip; uncovered samples are silence-padded.
Draw Things routing and planned continuous-song scene sources retain their existing separate contracts.
Director workflow upgrades are deferred.

Qualification used the same English sample/script with both Fish formats and both Qwen sizes;
Qwen's two reference modes and Fish synthetic mode also completed. Local English recognition
recovered the requested sentence in all seven checked takes. Five-second H3 and LTX 2.5 movies
completed using combined voice/music drivers at 1280×768 and exported with exact five-second stereo
audio. The H3 Turbo run used the existing headless Ref2VA path; this does not expand the ordinary
Studio Turbo picker's supported tasks.
This verifies execution and wording for those samples; it is not a general voice-similarity,
multilingual quality or lip-sync guarantee. See [implementation status](../STATUS.md) for limits.

## Native YuE2 music

Open **Movie → Generate Music…**, or **Generate Music** in the asset/track inspector. YuE2 runs
inside WeeTodd's native MLX engine; Studio does not install or invoke another YuE package.
Choose an existing compatible merged MLX checkpoint folder, or **Download verified 8-bit model…**
to install the pinned 4.53 GB model, tokenizer and stereo decoder into your model library.
This initial integration supports the merged layout published by
[npario](https://huggingface.co/npario/YuE2-3B-MLX). Split AR/NAR layouts, including the supplied
vanch007 conversion, are rejected with a layout error. Precision selects compatible files; it does
not quantize or convert a model. The upstream model/decoder weights use **CC BY-NC 4.0**,
separately from WeeTodd's source-code license.

Use genre, mood, energy, instruments, vocal character, language and requested BPM to build a
visible style prompt. Custom text is allowed in these categories. Enter lyrics with section markers,
or select Instrumental and describe the arrangement. These are model guidance, not guaranteed musical
constraints. **Quality** uses 32 acoustic midpoint steps; **Fast** uses 8. The length budget limits
semantic tokens at 25 frames/second and may truncate a song; it is not an exact requested duration.
Natural endings can arrive earlier, and truncated takes are labelled for review.

Advanced generation exposes Full/Melody/Direct composition, independent score and music samplers,
guidance, acoustic steps, seed, checkpoint precision and memory policy. Import/edit an ABC score,
or **Compose score only** before generating audio. **Resynthesize take** retains a selected take's
composition and music tokens while applying the current acoustic steps and seed. **Decode saved
latents** reruns only its stereo decoder. Saved stage files and checkpoint identities are checked
before reuse. The default memory policy releases the transformer before loading the VAE; retaining
components applies only within the current job. No background model cache remains after a job.

Every take saves a 48 kHz stereo PCM master plus its request, score, tokens, noise, latents, timings
and integrity metadata. Audition it, then **Add to Music track** at the playhead. Default placement
uses a track that mixes with source audio; an explicitly selected replacement track retains that
track's chosen behavior. Trim, gain, fades, mute and solo use the existing timeline controls.
**Collect Media** includes the stage artifacts while leaving shared model weights in place.

For audio-driven video, select a supported native video clip, open Music, select a take and set
its song in-point. **Use as audio driver** selects A2V. LTX 2.5 links the original song and its
selected interval so consecutive intervals can drive a continuous scene; other native models
prepare a lossless stereo excerpt covering their resolved frame duration. Source ranges are
checked before extraction.
The original song stays intact. When using the master on the Music track, optionally mute the
generated clip audio to avoid doubling it, and align the matching song region on the timeline.
The existing video preparation/review/render flow remains the next step. To plan multiple shots
from a song, use [Create a music video](#create-a-music-video), then review and explicitly apply
the shot plan.

**Export job…** saves the same request used by Studio. Headless music commands share the engine:

```bash
python scripts/studio_bridge.py music-generate --request music-job.json --output /new/take
python scripts/studio_bridge.py music-plan --request music-job.json --output /new/score
```

`music-resynthesize` accepts `source_artifacts` plus optional acoustic `steps` and `seed`;
`music-decode` accepts `source_artifacts`. Both require a fresh output folder. `music-inspect`
validates the model layout without loading its tensors. Unsupported settings fail explicitly.
Real generation qualification currently covers the pinned 8-bit checkpoint on M3 Ultra, including
a 159.8-second song at 32 steps in 123.2 seconds. This is one run, not a cross-device performance
guarantee. BF16/4-bit layouts have structural checks but are not real-render qualified here; M5
numerical behavior and lower-memory Macs need separate qualification.

## Build and open

Save your work and quit Studio before rebuilding its app bundle. The packager refuses to replace
a running copy: replacing an ad-hoc-signed bundle while it is open can make macOS reject the file
picker, causing a beachball followed by no dialog. If this happens after an older build script
replaced the app, quit normally and reopen the updated app before retrying the picker.

Run these commands from the repository root on Apple Silicon with **full Xcode 26 or newer**.
Standalone Command Line Tools are insufficient. The MetalFX frame interpolator requires a
macOS 26 or newer SDK:

```bash
python3 scripts/build_studio_app.py --configuration release
open "studio/.build/WeeTodd Studio.app"
```

The build resolves the Swift compiler, macOS SDK and platform macro plugins from one Xcode
installation. It compiles a small `@State`, `@Binding` and `@Observable` probe before building
Studio, so missing SwiftUI macros produce an early toolchain error. Shell overrides for a
separately installed Swift compiler or SDK do not override the selected Xcode.

Selection order is `--xcode`, then `DEVELOPER_DIR`, then `xcode-select -p`. For example:

```bash
python3 scripts/build_studio_app.py --configuration release --xcode /Applications/Xcode.app
```

This does not change the system-wide developer directory. A single Xcode installation is enough;
keeping multiple versions is optional. The argument also accepts an Xcode `Contents/Developer`
directory. Open a newly installed Xcode and complete its component setup before building.

**Xcode 27:** Apple changed SwiftUI's `@State` to a macro, which requires the matching platform
plugin. The build supplies that plugin search path explicitly. Updating Xcode does not change
Studio's macOS 14 deployment target or require a Swift 6 language-mode migration. The last
completed release qualification used Xcode 26.4; full Xcode 27 build and test qualification is
pending an installed Xcode 27 toolchain. See [Apple's State documentation](https://developer.apple.com/documentation/SwiftUI/State)
and [Xcode system requirements](https://developer.apple.com/xcode/system-requirements/) for the
build host requirements of the Xcode version you install.

The GUI supports macOS 14 and newer. Actual model/runtime requirements can require newer macOS.
MetalFX frame interpolation additionally requires macOS 26 and a supporting GPU. The helper checks
hardware support. The build script makes an ad-hoc signed application containing the Swift GUI,
`WeeToddCLI`, `StudioMetal`, the renderer source, and the runtime dependency lock. Building the
app does not install Python dependencies or copy models. All packaging/runtime preflight code is
tracked; no local agent skills are required. Packaging starts with a fresh bundle, verifies its
signature, and replaces the previous app only after success, removing obsolete bundled source files.
The app icon is bundled from `studio/Resources/AppIcon.icns`; `AppIcon.png` retains the supplied
artwork with only the outer white corners removed. The icon contains transparent standard and
Retina sizes.

In **Studio Settings**, choose **Set Up Managed Renderer** to download private Python 3.12.13 and
install pinned, hash-verified native dependencies. The installer verifies arm64, the project Python
requirement, package consistency, and Metal availability before activating the new runtime.
Each installation gets a new directory; prior runtimes remain available. The developer environment
and other applications' environments are preserved. There is no Linux virtual machine.
The bootstrap uses a pinned, checksum-verified uv release from Astral.

Advanced users can connect an existing compatible WeeTodd repository and Python environment.
Use guided model setup below, or import existing `weetodd-headless-v2` recipes to identify model
component sets. New clips select **WeeTodd (local)** or **Draw Things**, then a model and task;
automatic component selection inspects
compatible recipe contents. FFmpeg/FFprobe and
optional RIFE remain separately configured tools; a retail installer must package these tools
with their licenses and complete clean-Mac qualification.

Runtime settings, autosave, global assets, recipes, previews and jobs live under
`~/Library/Application Support/WeeTodd Studio`. User media and model weights stay in their existing
locations. `WEETODD_STUDIO_DATA` selects a separate data directory for isolated development tests.

## LTX 2.3 single-pass text-to-video

In Model setup, select **LTX 2.3 · Text to video · Single-pass distilled 1.1**. Import the existing
MLX distilled 1.1 bundle and a local Gemma 3 12B encoder, create the recipe, then choose **Use Recipe
for Selected Clip** on an LTX 2.3 T2V clip. The recipe uses eight evaluations, Shift 5 and staged
unloading with streamed weights. The inspector exposes editable Steps and Shift, fixed CFG 1,
and no refinement pass. Existing two-stage recipes keep their behavior.

This route creates synchronized video and audio directly at the clip dimensions. It currently
supports T2V; use the existing recipes for image, reference, audio conditioning and extension.
Exported clips and movie jobs preserve the same mode and settings. No model copies are needed.
See the [single-pass documentation](../README.md#ltx-23-single-pass-distilled-11) for model layout,
ComfyUI usage and measured validation. Physical 36 GB qualification remains open.

## Reuse local Draw Things H3 models

In **Studio Settings → Model setup**, select **MiniMax H3 · Draw Things models · Text to video**.
Scan your Draw Things model folder or import these existing files individually:

- H3 transformer: the supported `minimax_h3_i8x.ckpt` layout.
- Qwen encoder: the supported `qwen_3_vl_32b_50_i8x.ckpt` layout.
- Video/audio VAE: the supported `minimax_h3_vae_f16.ckpt` layout.
- H3 tokenizer folder: reuse one already installed, or use the small tokenizer download.

Names are examples; setup checks the tensor inventory, shapes and supported codecs. **Create Recipe**
creates metadata references next to the recipe. Choose an **H3** clip, **Text to video**, and this
recipe. This uses the native renderer and your local DT weight files; no DT connection or CU is
involved. A **Draw Things** clip still selects the separate gRPC/cloud route.

Original weights remain read-only and are decoded only in memory. No converted checkpoint or
persistent weight cache is created. Keep the generated `h3-dt-components-*` directory beside its
recipe. Model files and tokenizer remain in their original locations and must be available on the
machine executing an exported job. Existing H3 ComfyUI Component Loader fields can use the recipe's
component paths, including the original transformer file and generated metadata directories;
choose task `t2va`. The combined legacy pipeline loader is not this setup route.

This first release is experimental and limited to text-to-video with generated audio. Other DT
model families, first/last/reference frames and audio inputs need separate adapters/qualification.
LoRAs, resident execution and accelerators other than verified MPP projections have not been
qualified with DT weights. Use the checkpoint's default paging. Performance
and memory differ from the DT app; the VAEs currently execute in FP32. The transformer's packed
weights now decode and reorder on Metal automatically, retaining the previous native weight values.
Existing DT-weight recipes use this improvement without reimporting models. The matched 512×512,
124-frame, 19-evaluation run fell from 15:25 to 12:05 with a byte-identical movie, and process peak
fell from 18.07 to 16.35 GiB on M3 Ultra/256 GiB. Transformer MLX peak rose by 0.37 GiB; overall
MLX peak was unchanged. A physical 36 GB hardware test remains outstanding.

New DT recipes with normal working memory use **Automatic** projections. The renderer checks GPU
and macOS support, compares each new projection shape with standard MLX on first use and falls
back if verification fails. Lower-memory setup retains standard MLX. Existing recipes remain
unchanged; choose **Automatic** in the clip's projection control to opt in, or **MLX** to disable it.
For exported recipes, the equivalent field is `config.projection_backend` (`auto` or `mlx`);
ComfyUI offers it on **H3 Generation Config**. This does not enable resident loading or alter steps,
precision, memory chunk sizes or conditioning.

The matched normal-memory M3 Ultra run with Automatic projections completed in **10:51**, versus
**12:05** with standard MLX: **10.1% less total time**, with byte-identical video/audio. Process peak
remained **16.35 GiB** and MLX peaks were unchanged. This qualifies the measured configuration,
not lower-memory settings or other hardware.

Existing DT recipes also receive bounded read-only transformer payload mapping and compact video
decoder storage automatically. The video decoder keeps the original F16 weights, uses FP32
activations, and completes one block at a time to bound temporary memory. It does not load the
unused encoder. No new model download, conversion, permanent weight cache or settings change is
required. Other native model loaders and DT server/cloud clips use their existing paths.

The matched 512×512/124-frame/19-evaluation M3 Ultra (256 GiB) render now peaks at **9.48 GiB
process footprint**, down from 16.35 GiB, with a byte-identical video/audio MP4. Total time was
**10:50 versus 10:51**, effectively unchanged; this pass improves memory rather than overall
speed. Peak MLX allocation fell from 12.53 to 6.58 GiB. These measurements use fresh renderer
processes and fresh prompt caches, without clearing OS file caches, and do not qualify 36 GB Macs.

Cached-modulation DT blocks now prepare as one GPU batch, avoiding a wait after each tensor while
preserving the same weight values and sampler. Fixed weights and initial modulation preparation
remain eager. This is part of the shared renderer, requires no new model download or settings
change, and retains no weights after a block is released. Batch preparation timing is reported
separately from tensor decode submission timing.

The matched M3 Ultra run with batching completed in **10:13 versus 10:50**, with byte-identical
video/audio, unchanged MLX peaks, and effectively unchanged process footprint (**9.51 versus
9.48 GiB**). Block preparation fell from 109.4 to 91.7 seconds. The observed 5.7% total-time
reduction is from one desktop comparison; unchanged block computation also ran faster, so batching
does not account for every second saved. Physical 36 GB qualification remains open.

Normal-memory DT recipes with **Automatic** projections can now prepare one decoded transformer
block ahead while the current block computes. The renderer enables this only for its resolved
MPP path, one-block paging, cached modulation and no retained-page cache. It uses about 0.72 GiB
for the additional block plus temporary preparation buffers. **MLX** projections or **Lower memory**
disable it. It creates no converted model files and releases the worker and prepared weights on
unload, failure or cancellation. Exported headless jobs and composable ComfyUI nodes use the same
behavior. Render reports include lookahead hits, weight bytes and waiting time; overlapping
preparation/compute timings cannot be summed as separate elapsed stages.

The integrated 512×512/124-frame/19-evaluation M3 Ultra run completed in **9:07 versus 10:05**
(9.7% less time), with a byte-identical video/audio MP4 and all weighted stages unloaded. Process
footprint was **9.48 GiB** and overall MLX peak remained **6.58 GiB**; transformer MLX peak rose
from 5.31 to 5.91 GiB. The earlier same-workload prototype took 9:31, so elapsed gains vary.
This is desktop evidence for the tested recipe, not physical-36-GB or DT sampler parity.

The same setup is available from the CLI, without writing a recipe by hand:

```bash
python scripts/setup_models.py download h3-dt-tokenizer --destination /path/to/library
python scripts/setup_models.py create h3-draw-things-text \
  --component dt_transformer=/path/to/DrawThings/minimax_h3_i8x.ckpt \
  --component dt_qwen=/path/to/DrawThings/qwen_3_vl_32b_50_i8x.ckpt \
  --component dt_vae=/path/to/DrawThings/minimax_h3_vae_f16.ckpt \
  --component tokenizer=/path/to/tokenizer \
  --profiles-directory /path/to/recipes
python scripts/render_headless.py --recipe /path/to/recipes/created-recipe.json \
  --output-directory /path/to/output
```

Use the tokenizer folder returned by the download/scan and the recipe filename returned by setup.
The model files stay in the DT model store. The tokenizer download retains its pinned source terms.

## H3 INT8 finetunes

Supported Comfy H3 `int8_tensorwise` safetensors files, including Singularity Ref2VA v1.3 INT8,
can be selected as the transformer in the existing H3 setup presets. Keep the matching H3 task
manifest, Qwen3-VL encoder, processor/tokenizer and video/audio VAEs; those components are shared
in place. A finetune remains an ordinary H3 recipe in Studio, with the same conditioning and
generation controls. Tensor architecture validation does not establish a finetune's training-task
support: use the task recommended by its publisher.

The loader validates the complete file and quantization metadata, reverses supported ConvRot
encoding, and adapts attention layouts in memory. It reads the original file without creating a
converted checkpoint. The default retains fixed weights and one decoded transformer block;
preflight includes temporary decoding space. Explicit paging-cache budgets can retain decoded
blocks to reduce repeated reads, at increased RAM use. Staged unloading remains the default.
Native execution uses BF16/FP32 and does not reproduce Comfy's CUDA W8A8 activation quantization
or imply the same speed, memory use or output. This route remains experimental.

## Native H3 NNC core

In an H3 clip, open **Memory and execution → Transformer → Native NNC (experimental)**.
This is an explicit backend choice for an existing H3 model, not a separate model family.
The first qualified configuration uses a supported Comfy INT8 Ref2VA transformer, one complete
ComfyUI BF16 Ref2VA Turbo LoRA at strength 1 with contiguous QKV layout, **4 steps** in Studio
(`steps: 5` schedule points in a recipe), Euler sampling and Drop AdaLN. Use checkpoint-default
paged residency, no retained page cache, automatic chunk controls and auto/MLX projections.
The native core has fixed buffer policies; MLX chunk controls only affect MLX stages.
Unsupported settings fail rather than falling back silently. Cache accelerators, sparse/approximate
attention, control, continuation and refinement are not qualified for this backend. Packed input
is limited to 40,000 rows, including text and reference media.

The independently implemented Swift/NNC worker reads installed weights in place and keeps one
GPU block plus one CPU prefetch. It uses FP16 projections with FP32 residuals and attention.
MLX retains the shared conditioning, sampling schedule, input/output projections, previews and
VAEs. Progress includes each native block. Staged unloading stops and reaps the worker before
VAE decoding, including on failure or cancellation; a watchdog exits it if its parent dies.
Keeping the transformer warm is still an explicit shared-runtime control. Worker Metal allocation
and parent MLX memory are separate measurements and must not be confused with total process memory.

Release packaging builds this helper and includes pinned dependency notices. Development checkouts
can run `python scripts/build_h3_worker.py`. A saved full-size 39,967-row evaluation through the
integrated shared DIT path matched the qualified prototype across all 214,862,592 Float32 values,
with the worker reaped after completion. This validates integration numerics, not Draw Things
parity, complete video quality or performance across hardware. MLX remains the default.

## Guided model setup

For remote generation, see [Draw Things](#draw-things--experimental). Local model recipes below
configure the native MLX engines.

1. Open **Studio Settings → Model setup**. Built-in presets appear independently of installed recipe
   count once a native worker or the optional renderer is configured. Choose H3 text/image/reference
   or LTX 2.3/2.5 text/image. Swift LTX 2.5 also offers spatial DFR and one- or two-round
   temporal DFR presets, MSR images, Ingredients sheets, Union, Motion Track and CrossView
   controls when its worker is enabled. H3 also offers a native Fun ControlNet guide preset.
   Compatible installed adapters can be reused or acquired through pinned native catalog entries.
2. Choose **Set Up… → Use Existing Models**, select model folders (including an existing ComfyUI
   `models` folder), then scan. Inspection reads bounded headers and manifests, never model tensors.
   A single candidate is selected automatically; multiple candidates require your choice. Missing
   components have **Import…** and **Download…** controls beside their labels. Import links a local
   file/folder without copying it. Download selects a compatible package and opens its size, contents,
   source terms and destination for review; it does not immediately start the transfer. Components
   without a catalog download can still be imported. Validation explains incompatible architecture
   or task support.
   Swift H3/LTX 2.5 presets use the same folder scan or **Import…** for each installed
   component, with no Python process or weight copy. Their scan reads bounded metadata;
   the selected worker validates the complete stack before a text render or after clip media
   is attached for an image/reference render. H3 image/reference setup explicitly asks for
   a Qwen vision tower: choose installed vision-capable pages or a compatible raw
   Qwen checkpoint for that field. Text encoding itself requires the supported paged pack.
   Native H3 transformer/audio fields require compatible direct/folded-weight files;
   older Python paged transformers and unfurled audio bundles are excluded from native choices.
3. Use **Automatic** to select a lower-memory policy on Macs with 64 GB or less, **Lower Memory** to
   request supported memory-saving settings, or **Custom** to retain the preset policy for later
   advanced adjustment. Memory information is advisory and does not promise fit, allocate RAM, or
   enforce a hard limit. Clip size/duration and other resident applications still matter.
4. **Create Recipe** runs the shared component/configuration preflight and writes a new recipe.
   Image/reference presets still need media attached to a clip before full render preflight can pass.
   For a Swift text-to-video preset, the selected worker runs preflight before the profile is kept;
   for a Swift image/reference preset, attach media and prepare a clip for worker preflight.
   Existing recipes and model files are preserved. Automatic selection can use the new compatible
   components; a specific recipe can be pinned under **Advanced generation**. **Set Up Models…**
   is also available from missing-model actions.

**Download or prepare a model** shows compatible catalog items with source terms, download size and
required space before an explicit download. Prefer the
[H3 Q8 vision encoder](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-vision-paged) or
[LTX 2.5 distilled Q8 package](https://huggingface.co/Vayden/LTX-2.5-MLX-Q8-Paged) marked
**Preconverted (Recommended)**. Source conversion remains an alternative. LTX 2.5 source preparation
downloads the five official distilled components and
converts transformer/Gemma to paged Q8. H3 encoder preparation downloads just the compact Q8 encoder,
its support files and full Qwen architecture config, then retains the vision page.

With Swift H3/LTX 2.5 enabled, compatible preconverted downloads run in Swift through
URLSession. Transfers stream to disk, resume retained partial bytes, and verify exact pinned
size and SHA-256 before installing a new folder atomically. Verified installed files are
hardlinked on the same volume or linked across volumes; keep the original files available.
Cancellation preserves partial downloads without publishing an incomplete package. Progress
and **Cancel** appear in the setup window. The native path uses a Studio Keychain token or
`HF_TOKEN`; it never writes the token into download state, provenance or logs.
The merged native catalog has 19 pinned packages supplying mandatory component fields for ordinary H3/LTX profiles,
including direct H3 transformer/support, Qwen, video VAE, folded audio VAE and tokenizer, plus
supported task adapters. This field coverage does not establish every checkpoint/task combination;
source terms, gating and structural admission still apply to every file. H3 image/endpoint tasks
use the verified 64-curve BF16 FL2VA package; the full-width Singularity package serves its
supported text/reference/Fun Control tasks and is rejected for native image/endpoint generation.
The older Python paged-transformer/support packs and source conversion remain separate. The
folded audio VAE retains its source license, notices and conversion modifications. Installed-file
checks verify exact hashes, shared-file reuse and package discovery; they do not establish a
complete clean-machine download/install of all weights or universal adapter compatibility.

For the optional Python H3 setup, choose four downloads in the same library: the **text/image or reference Q8 transformer**,
its matching **support files**, the **Q8 vision encoder**, and the **Q8 video VAE**. The support files
include the official task manifest, audio VAE, tokenizer and processor. Downloads specific to another
task are hidden; the encoder and video VAE can be reused across tasks. After installation, scan the
downloaded folders, choose the components and create the recipe. No H3 weight conversion is required
for these prepared sets. Optional Turbo/control LoRAs remain separate.

Selected existing roots are checked for exact source checksums before downloading replacements.
Same-volume source files are reused through links; cross-volume source references remain linked,
with copies only where final component packaging needs them. Existing converted components can be
selected directly without conversion. Partial downloads resume after cancellation or network failure.
Sources are retained in the chosen library’s `.weetodd-downloads` directory for reuse. Only verified,
fully prepared output is installed; source terms/attribution remain with the output. Gated sources
require accepting their upstream access terms. In **Hugging Face access**, open the
token settings link, paste a read token and save it to macOS Keychain. The field clears after saving;
the token is passed only in the download process environment and never written to recipes or setup logs.
Remove it with the same controls. An existing CLI login (`hf auth login`) remains supported when no
Studio token is saved.
The setup log explains authentication, disk-space, checksum and conversion failures.

For CLI users, `python scripts/setup_models.py --help` exposes the same catalog, scans, recipe creation
and downloads. For example:

```bash
python scripts/setup_models.py download h3-qwen-q8-vision-preconverted \
  --destination /path/to/shared-models --existing-root /path/to/ComfyUI/models
python scripts/setup_models.py download ltx25-distilled-q8-preconverted \
  --destination /path/to/shared-models --existing-root /path/to/ComfyUI/models
```

Run the command for the model you need. These downloads skip local conversion. The catalog also
offers `h3-qwen-q8-vision` and `ltx25-distilled-q8` to convert from pinned source files.
After a download, scan the returned directory, choose the components, and create the recipe. ComfyUI
users can select those same paths in existing loaders; setup never downloads or converts during a graph.
The [headless example/schema](../examples/headless/README.md) remains available for direct JSON users.

### Model setup troubleshooting

| Symptom | Next action |
| --- | --- |
| LTX clip shows **0 recipes** | Choose its built-in preset in **Studio Settings → Model setup**, scan existing components or download the prepared package, then **Create Recipe** and **Use Recipe for Selected Clip**. Downloading weights alone does not create a recipe. |
| **Prepare Clip** rejects the official LTX 2.5 distilled LoRA with `to_gate_logits` targets | Update the renderer and prepare again. The validator now accepts the official attention-gate weights. Existing models and prompts can be reused. For an app-managed runtime, refresh it with **Set Up Managed Renderer** from the updated app. Preparation and generation alerts now show the renderer's specific error; **Show Log** keeps the traceback. |
| Built-in presets or preconverted downloads are missing | Use the updated app and renderer. Rebuilding the app updates its bundled source; an existing managed runtime keeps its old source snapshot. Choose **Set Up Managed Renderer** to install the updated snapshot, or connect an updated repository with a compatible Python environment. Existing media and models can be reused. |
| Download reports denied access / 401 / 403 | Open **Model source**, accept that repository's access terms with your Hugging Face account, and save a read token for the same account under **Hugging Face access**. Review the setup log for the exact failure. |
| Download was interrupted | Repeat the same download with the same destination. Verified files are reused and partial files resume. A completed package should instead be opened through **Use Existing Models**. |
| Setup reports insufficient space | Choose a library on a drive with enough free space. Preconverted downloads skip local quantization and its intermediate storage; the displayed download size is not a RAM estimate. |
| Scan finds several transformer candidates | Select the distilled transformer for the LTX 2.5 distilled preset. Common architecture headers alone cannot prove Dev/distilled training identity; the curated preconverted package removes that ambiguity. |
| H3 reference clip rejects a text-only encoder | Select the new **H3 Q8 vision encoder · Preconverted (Recommended)** package. The older v1 text-only export remains useful for T2VA but lacks the vision weights needed for image/reference conditioning. |

Setup creates component recipes, while **Prepare clip** validates the final clip's media and settings.
If a reference/image clip still needs attention, attach the required media and follow its Actions entry.
For a complete CLI walkthrough, see [download and create an LTX 2.5 recipe](../examples/headless/README.md#download-and-create-an-ltx-25-recipe).

## Clip generation controls

Choose **Generation → WeeTodd (local) / Draw Things → Model** in the clip inspector. Local models
include MiniMax H3, LTX 2.3 and LTX 2.5. Available tasks and controls appear directly, without a
required template or preset step. Tasks come from the installed compatible
model recipes. Image to video requires a first image; First + last frame requires both endpoints.
Changing tasks preserves attachments and names any conflicting or missing input. A reference-only
H3 recipe cannot make a text-only clip appear ready. Model, render size and seed are available in
the ordinary generation flow. Switching providers restores the last local model and its saved
settings. Choosing a task retains the selected model; incompatible
combinations explain what needs changing. Missing prompts or frame inputs do not hide valid sampling
controls. Complete media and model preflight still runs before generation.
An older clip with a missing or task-incompatible pinned recipe offers **Use automatic model
components**, preserving its media and explicit parameter edits.

Changing a shot in a continuous LTX 2.5 scene to H3, LTX 2.3 or Draw Things separates that shot
for independent generation. Remaining LTX scene groups keep their shared sound and boundary-image
settings. Images, accepted takes and timeline edit points are preserved; **Undo** restores the model
and scene connections together. An older saved project with an incompatible scene connection offers
**Separate this shot** in Clip Continuity. Native continuation state cannot be shared across models.

The inspector displays sampling controls and compatible LoRAs/groups with editable strengths.
Ordinary native H3 Euler **Steps** means actual evaluations: 19 evaluations correspond to 20 stored
schedule points. Specialized or fixed schedules explain their restrictions. LTX stage-one and
refinement controls are separate; fixed distilled schedules remain fixed. Native H3 uses distilled
guidance and fixed video/audio shifts; the panel explains built-in guidance without offering inactive
CFG or Shift fields. Supported native
LTX CFG and Draw Things configuration values remain editable. Unsupported submitted overrides fail
validation instead of being ignored.

Edited controls show **Modified**; **Reset** removes those overrides. Optional execution presets and
custom recipes live in **Advanced generation**. Existing clips preserve their imported recipe
settings. Choosing an advanced preset opts into the new explicit
selection. **Generate** prepares and validates the clip automatically. The separate preparation and
prompt/settings review remain available. Clip/movie headless export uses the same resolved recipe.

### Acceleration settings

App-level H3 acceleration preferences are separate from creative sampling settings. Automatic
projections use the existing hardware-qualified backend and numerical fallback checks; **MLX**
explicitly uses the standard backend. Automatic memory selection uses the lower-memory policy on
Macs with 64 GiB or less, and retains the recipe policy on larger Macs pending further qualification.
The detected RAM figure is advisory, not available RAM or a hard allocation limit.

Each explicit clip can override those preferences. **Paged** selects lower-memory execution with
the checkpoint's existing layout; it does not convert a resident checkpoint into pages.
**Paged · larger workspace** keeps that layout but uses normal working buffers, allowing a separate
comparison of working memory and weight residency. This option is not qualified for 36 GB hardware.
**Resident**
retains all transformer blocks during sampling, requires normal memory mode and a zero page-cache
budget, and is intended only for ample-memory systems. The transformer still unloads before VAE
decoding, including failure/cancellation cleanup. This is separate from keeping weights warm across
jobs. Existing Custom clips do not silently inherit newly changed app preferences.

Balanced and Speed currently preserve the recipe's sampling schedule. Speed is not a promise of
fewer evaluations or a measured speedup; experimental approximations are not silently enabled.
Low memory selects the supported lower-memory policy. Dimensions, duration and attached media
remain explicit clip choices. Hardware-specific speed recommendations require matched measurements.

### Matched H3 execution measurements

On an M3 Ultra with 256 GiB unified memory (2026-09-10), a saved 512×512, 124-frame,
24 FPS H3 T2V recipe used 19 Euler evaluations, seed 42, Q8 paged FL2VA transformer,
paged Qwen and Q8 video VAE. Prompt, models, quantization, dimensions and schedule were fixed.
Each run used a fresh renderer process; the conditioning cache could reuse encoded text, so compare
sampling separately from whole-job time. No other generation ran concurrently.

| Execution | Total | Sampling | Video decode | Process RSS peak | MLX stage peak |
| --- | --- | --- | --- | --- | --- |
| Paged · lower memory, MLX (saved baseline) | 1065.8 s | 1016.1 s | 35.2 s | 5.69 GiB | 6.64 GiB |
| Paged · lower memory, Automatic | 1046.4 s | 1010.0 s | 34.6 s | 5.62 GiB | 6.64 GiB |
| Paged · larger workspace, MLX | 574.7 s | 535.5 s | 37.9 s | 6.34 GiB | 6.98 GiB |
| Resident, MLX | 535.1 s | 493.8 s | 38.5 s | 31.18 GiB | 32.32 GiB |
| Resident, Automatic | 558.6 s | 514.9 s | 41.0 s | 31.18 GiB | 32.32 GiB |

All five movies were byte-identical, with 124 video frames, stereo 32 kHz audio, 8.3 ms
A/V drift and every weighted runtime released. These are individual runs, not a statistical
backend ranking. Automatic projections passed the hardware/numerical checks but showed no useful
speed gain here. The resident policy roughly halved sampling time, with a substantial memory cost.
The larger-workspace paged run achieved most of that gain with only a small measured peak increase,
showing that chunking/working-buffer policy accounts for much of this baseline's slowdown.
Full residency saved another 39.6 s overall while adding about 25 GiB to the MLX stage peak.
Try **Paged · larger workspace** first for this recipe before full residency. Normal memory mode
also selects a larger decode batch; decode did not improve in these runs.
Process RSS and the largest instrumented MLX stage are distinct counters, not additive RAM totals.
This is evidence for this recipe on this Mac, not qualification for a 36 GB Mac or every task.

## Progress, measurements, and H3 page retention

Native renders now send live stage/evaluation updates to the status bar. Sampling shows its own
progress, followed by decoding/publication; the bar is not an estimated whole-job percentage.
Elapsed time and the age of the last renderer output remain visible during long steps. A quiet
interval alone does not mean the renderer has stalled. Cancellation remains attached to the job.

New render versions retain measured render time and available memory/timing statistics. Select a
version to see its summary or expand **Versions** to compare takes. Older projects still open;
historical versions without saved statistics show no invented measurements. **Process peak** is
renderer RSS, while **Instrumented stages** and **MLX generation** describe different MLX counter
scopes. Do not add those peaks or treat them as total system RAM. LTX **Pre-decode pipeline** time
includes encoding, model loading, sampling and latent upscaling; H3 reports transformer sampling.

For H3, **Advanced generation → H3 page cache** offers Recipe default, Off, or 4/8/12/16 GB.
The recipe setting is `config.paging_cache_gb` (0–16 decimal GB; default 0). This is extra retained
transformer weight memory, not a limit on total generation memory or a promise of fit.
Start with Off versus 4 GB using identical prompt, seed, dimensions, schedule and storage. Compare
the measured process/MLX peaks as well as wall time; return to Off if memory pressure increases.
The cache retains native checkpoint pages in their stored precision; direct Draw Things and Comfy
INT8 sources instead retain decoded BF16/FP32 blocks, counted at their decoded size. It releases
retained weights after each sampling run, including cancellation/failure. It requires a paged or
supported direct transformer and cannot be combined with full block residency. A block larger than
the remaining budget is bypassed.

ComfyUI exposes the same setting through **H3 Paging Settings (Experimental)** between Generation
Config and the composable H3 Sampler. Headless movie/clip jobs carry the selected value. H3
`result.json` includes paging counters, retained peak/budget bytes, and loading/setup/compute timing.
Cache counters cover the configured run; other pager counters identify their executor-lifetime
scope. File-load calls are not measurements of physical disk reads. The cache avoids some reloads
but does not remove module construction or attention computation. Tiny FP32/Q8/LoRA tests establish
parity and cache behavior; a full-size speedup on a 36 GB Mac has not yet been measured.

Head and FFN chunk controls now reach newly loaded paged blocks. Previous automatic/lower-memory
comparisons could therefore exercise effectively identical transformer settings. Rebaseline after
updating: smaller chunks can reduce workspace but add dispatch overhead, so the fix alone is not
a speedup claim. Query chunking and the retained-page budget are separate controls.

## LTX 2.5 images, controls, and references

See [reference inputs by purpose](#reference-inputs-by-purpose) for the complete input matrix.

For ordinary image-to-video, choose **WeeTodd (local) → LTX 2.5**, import/select an image,
and choose **Use in clip → First frame · Image to video**. Prepare the clip after attaching it;
compatible installed components are selected automatically. The Reference role selects MSR
conditioning and requires its dedicated compatible adapter components.

Guided setup now also offers **IC-LoRA control**, **Ingredients reference sheet**, and **MSR image
references**. Import the corresponding dedicated adapter in addition to the existing model
components. Setup checks its tensor headers and compatibility; a style LoRA cannot substitute.
These recipes use the existing distilled full-resolution single-stage renderer. They are separate
from the basic image preset and do not require a spatial upscaler for that single-stage route. With **Use Swift MLX for LTX 2.5** enabled, Model Setup offers
**LTX 2.5 · MSR images · Swift** and **LTX 2.5 · Ingredients sheet · Swift**.
Their folder scans inspect bounded adapter headers and link weights in place. Prepare runs
shared worker preflight after the described images are attached. MSR keeps image order,
with one optional background last, and maps role, priority, frame count, sizing, reference
strength and attention strength. Ingredients uses one described sheet and at least 121 frames.
Both run eight full-resolution evaluations; there is no second stage or spatial upscaler.
Ordinary LoRAs, DFR and audio drivers cannot be combined with these single-adapter profiles.
Ingredients setup admits compatible 2.5 full-resolution rank-128 sheet adapters after complete
header/shape checks and starts them at strength 1.0; legacy 2.3 starts at 1.2. The pinned official
2.5 download is gated, and actual header qualification with the available access remains pending.
The current MSR V1 contract has 480 video pairs and five visual-slot tensors. The actual MSR V2
header has 1,152 pairs, extra audio/video-to-audio targets and five audio-slot tensors; V2's
sparse-speaker/reference-audio contracts are not implemented and its files are explicitly rejected.
Installed-app five-second 512 × 256 jobs passed generation, two decoded previews, acceptance
and reopening with Python unavailable. Their render-plus-acceptance times were 83.49 seconds
for two-image MSR and 99.38 seconds for Ingredients; worker process peaks were 4.03/4.00 GB,
excluding FFmpeg. The Ingredients result still duplicated subjects, so broad identity and
audio quality remain unqualified. Saved recipe-backed ComfyUI MSR and Ingredients graphs
produced byte-identical movies to these Studio takes. MSR additionally completed the new
native exported-job CLI route, assembled its movie, reopened its accepted project and reused
its take on resume; Ingredients native exported-job inference remains unqualified.

- **Control:** attach a preprocessed guide video as Control and choose its matching guide type
  (such as depth, pose, motion tracks or crossview). The recipe's adapter must support that type.
- **Ingredients:** attach one image using **Appearance · Ingredients sheet**, use at least
  121 output frames, and describe the sheet and intended scene in the prompt.
- **MSR:** attach one to five images as **Appearance · MSR image**, describe each, and
  choose subject/object/clothing/background, priority, reference frames, sizing and attention
  strength in Conditioning. Only one background is allowed. Recipe default preserves matching-image
  options from an imported recipe; removed attachments never leave hidden recipe media active.

These routes reuse shared renderer validation and remain subject to adapter and memory constraints.
First-frame success on 36 GB does not establish MSR/control memory fit or reference quality.

## LTX 2.5 Ripple Director

Ripple restyles an existing video using edited versions of its source frames. Configure
**Runtime Settings → LTX Ripple video restyling** with an installed
[LTX25_Ripple_v11.safetensors](https://huggingface.co/WepeNerd/LTX-Ripple) adapter and
compatible LTX 2.5 distilled components. Automatic selection uses a compatible installed
profile. Models are runtime settings shared by clips. Studio validates adapter identity
and component compatibility before loading weights; it uses the shared local MLX renderer.

1. Select a video clip and open **LTX 2.5 Ripple → Open Ripple Director…** in its inspector.
2. Edit the required first-frame reference. Use **Edit with Draw Things…** to open the
   image workspace with the extracted source frame on its canvas, or import an edited image.
3. Scrub the full-width timeline and choose **Add reference at playhead…** for additional
   frames. Each reference targets a distinct source frame; there are at most nine total.
4. Click a timeline thumbnail to reopen its editor. Right-click and choose
   **Delete reference frame clip** to remove an additional reference. The first-frame slot
   stays at frame zero; replace its image to change it.
5. Adjust the prompt, resolution, seed, **Ripple LoRA strength**, and audio policy, then
   generate a new take. Preview it and explicitly apply it when satisfied.

The IC-LoRA strength starts at **1.35** and is editable. The prompt starts with the
[author's workflow instruction](https://huggingface.co/WepeNerd/LTX-Ripple#prompting)
for preserving source motion, timing, camera movement, composition and unchanged content
while propagating the first-frame edit. Add a brief description of your visual change
when needed. The compact clip inspector and Director edit the same saved settings.
With native LTX selected, Studio inspects the source interval and extracts edited-frame
canvases in Swift. **Use Swift MLX for Ripple (experimental)** in Ripple Runtime Settings
also enables native preparation and generation. This route freezes edited images, streams
the causal RGB24 guide, encodes it in bounded VAE tiles, uses the pinned author LoRA in
an eight-step single-stage sampler, decodes live frame previews, preserves source audio
when selected, and verifies the published editorial frame count before saving a take.
The default remains the Python-backed LTX 2.5 route. A 64 × 64, 10-frame silent native
worker test produced a decodable take in 39.04 seconds with a 1.575 GiB peak MLX allocation.
Real source-guided tests also completed at 768 × 448 for three seconds (72 editorial frames,
163.34 worker seconds, 3.09 GB peak MLX, 5.07 GB worker footprint) and 1152 × 768 for five
seconds (120 editorial frames, 865.417 worker seconds, 9.44 GB peak MLX, 11.11 GB worker
footprint). Both fully decoded and sampled frames showed a coherent kitten. These are two
specific sources and recipes, not general subject/motion or timed-anchor quality qualification;
footprints exclude external FFmpeg. Source-audio mux passed a separate short test.
A separate installed-worker Studio lifecycle test ran with Python unavailable: it prepared
the three-second kitten source using an aspect-preserving center crop, rendered 72 frames,
delivered live previews, applied the take, and reopened the saved project. The worker took
151.12 seconds with 3.09 GB peak MLX and 5.22 GB peak process footprint. The earlier direct
test guide resized the source vertically, so its identical settings do not imply identical
output pixels. Broader source motion and production quality remain unqualified.
A further three-second 768 × 448 worker run combined a frame-36 reference and source audio.
It completed in 182.60 seconds with two live previews, a 3.21 GB MLX peak and a 4.84 GB worker
footprint (FFmpeg excluded). The decoded audio kept source timing and correlated at 0.999995.
The middle reference showed the original gray cat while the edit prompt requested a white cat;
the white output did not land on that gray reference. This establishes execution and audio
preservation, not timed-anchor fidelity. Keep additional anchors experimental.

Frame numbers are relative to the clip's captured trim interval. Studio adopts the source
frame rate and extracts matching frames without resampling. Use a constant-frame-rate source;
variable-frame-rate clips are rejected before inference to preserve motion and audio timing.
The first edited frame primes
the IC guide; additional references use native timed image conditioning. Multiple references
are an experimental Studio extension, not an author-qualified nine-frame workflow.
The render uses eight full-resolution Euler steps with the distilled checkpoint.

**Preserve source audio** keeps the source interval's soundtrack, re-encoded for the
published movie; generated audio is discarded. Silent source clips are valid. Choose
**Silent output** to omit audio. New takes retain their inputs and receipts, preserve the
original source until explicit application, and remain reviewable when settings change.
Collect Media includes the source, references and retained takes. To restyle a changed timeline
source or trim, start a new draft and reassign references; the existing draft retains its original
interval.

## Reference inputs by purpose

Choose what the model should retain from an asset's **Use in clip** menu. Selecting a purpose
sets its task in the same undoable change. Other attachments stay visible; preparation reports
incompatible combinations instead of dropping inputs. Role and guide menus filter by media type.

| Input purpose | Local H3 | Local LTX 2.3 | Local LTX 2.5 | Draw Things video |
| --- | --- | --- | --- | --- |
| Image appearance | Ref2VA image | Ingredients sheet | MSR images or Ingredients sheet | H3 Ref2VA images |
| Movie appearance / story | Ref2VA movie context | Sampled Ingredients sheet | Sampled Ingredients sheet | Not exposed |
| Movie motion / composition | Guide + Fun ControlNet | Guide + Union IC-LoRA | Guide + Union IC-LoRA | Not exposed |
| Audio-driven video | Ref2VA timing reference; new generated audio | Supplied audio is frozen | Supplied audio is frozen | Not exposed |
| Sound / voice reference | Ref2VA audio with visual context | Use Audio driver | Use Audio driver | Not exposed |
| Endpoint image | First/last frames | First/last frames | First/last frames | H3 FL2VA endpoints; LTX first frame |

**Appearance / story · make reference sheet** samples six movie frames into a labelled image.
Review that asset and describe the subjects, setting and intended story in the prompt. It provides
visual context; it does not infer a screenplay, preserve movie timing or copy the soundtrack.
**Motion / composition · make edge guide** runs Canny extraction and saves a silent guide movie.
It crops to the shot's aspect ratio and stops at the shorter of the source and shot duration.
It never loops the source to invent extra motion. Preparation uses 24 fps; custom-rate guides
can be imported as already prepared controls. Limits are 60 seconds, 2048 pixels per dimension
and 3600 frames. Cached outputs are verified before reuse. Originals remain unchanged;
derived assets and attachments undo together.

For Ingredients on either LTX version, the attachment description is included in the resolved
reference-sheet prompt. The shot prompt remains editable as written; review the complete resolved
prompt before generation. An already structured Ingredients prompt is preserved.

LTX 2.3 Model Setup now offers **Ingredients reference sheet** and **Union IC-LoRA motion guide**.
Ingredients requires a Dev bundle with its distilled helper adapter, one image, 768×448,
at least 121 frames and 24 fps. Union uses a distilled bundle and dimensions divisible by 128.
Both require resident loading without generic LoRAs. LTX 2.5 has dedicated Ingredients, MSR
and control setup choices. Automatic selects compatible installed components. IC-LoRAs stay
outside style LoRA groups because they change reference encoding and stage behavior.

For Draw Things, select **Image references · H3**, refresh the connection, select an H3 **Ref2VA**
model, and attach 1–9 still images. This requires the updated transport helper. FL2VA models
cannot reinterpret those references as endpoints. Other video models retain their discovered
endpoint capabilities. LTX IC-LoRAs and movie/audio references have no qualified mapping in
the current helper. Software and transport tests do not establish real-model quality or memory fit.

## H3 reference clips with paged Q8 models

Import a recipe produced by `scripts/prepare_h3_reference_recipe.py`, choose **WeeTodd (local) → H3**
and the reference task, and attach images with the **Reference** role. Compatible components are
selected automatically; pin the imported recipe under **Advanced generation** if needed.
The shared renderer uses genuine
Ref2VA Q8 transformer pages and vision-capable Qwen v2 pages. Start with one image, five seconds,
640×384 and the recipe's 19 dense evaluations; add a second reference only after checking memory.
The existing text-only Qwen page export cannot encode reference images.

See the [model preparation commands](../README.md#experimental-h3-reference-paging). Choose the
[preconverted Q8 vision encoder](https://huggingface.co/Vayden/Qwen3-VL-32B-H3-MLX-q8-vision-paged)
in guided setup, together with **H3 reference transformer Q8**, **H3 reference support files** and
**H3 video VAE Q8**. Alternatively, prepare your own files with the bounded-memory conversion
commands. Select the genuine Ref2VA transformer; the text/image transformer is a different model.
Clip/movie headless
export preserves this recipe so Studio can be closed during generation. One-image 640×384 generation measured a 21.70GB complete Comfy process peak on an M3 Ultra
with 256 GiB. The headless output was byte-identical and peaked at 21.26GB.
A 36GB physical-device maximum is not yet established; the header-based estimate omits reference-dependent workspace.

## Editing

- Clip inspector at upper left, inherited movie settings below; central viewport and timeline;
  collapsible Global, Project, and selected Clip asset stores at right.
- System, Light, and Dark appearance. Section colors in the design wireframe are not used.
- One main video track, titles, and additional named audio tracks with mute, solo, and source-audio
  replacement. Set each region's start, source in, duration, volume and fades in its inspector.
- H3, LTX 2.3 and LTX 2.5 generated clips, imported movies/stills, and image sequences. Sequence import
  uses the movie frame rate, natural filename order, linked original frames and a ProRes editing proxy.
- Drop a movie asset on the timeline to create a clip and a source reference in its Clip Assets.
  Generated clips expose **FF** (First Frame / I2V) and **LF** (Last Frame) slots at their timeline
  endpoints when supported by the model and recipe. Drop one image from any asset store or Finder
  onto a slot, or click an empty slot to import. Filled slots show thumbnails; another drop replaces
  that endpoint, and the context menu removes it. Assignments are undoable, link the original into
  Clip Assets, update the inspector/task, and invalidate prepared generation. Draw Things H3 offers
  both slots (last requires first); Draw Things LTX currently offers first only. Movie/still clips
  have no generation slots. Drop into a native generated clip's body for an interior keyframe.
  Drag clip cards to reorder them.
- Split rendered/imported clips, duplicate, edit source in/duration, and create extension clips.
  Before-extension is offered for LTX 2.3. Extension uses the referenced source movie as model context;
  trimmed timeline boundaries are not currently extracted into a new extension context automatically.
- **Insert Bridge to Next Clip** extracts the selected clip's last visible frame and the next clip's
  first visible frame. It creates an LTX 2.5 shot with linked first/last anchors and opens its prompt.
- The prompt editor fills the window. **Prepare clip** validates model paths, task support and inputs,
  then displays the exact resolved prompt. **Generate clip** executes that prepared recipe.
  H3 reference/audio/extension tasks currently require their complete native six-section prompt.
- Automatic task selection and the selected recipe's compatible adapters preserve the existing
  backend contracts. Controls expect preprocessed guide media. Missing or incompatible inputs fail;
  the editor does not guess an arbitrary ControlNet, LoRA, pose extractor or reference description.
- Generated versions remain in Clip Assets. Undo/redo and autosave protect edits. Collect Media writes
  a separate project with relative media references, including used Global assets. It preserves shared
  model/LoRA paths; it does not package weights or rewrite model recipes for another machine.
- Seed typing is grouped into one Undo step per editing session. Command-Z and Shift-Command-Z
  restore the complete previous/next seed, including while the field retains focus. Unchanged field
  writes do not add Undo steps or clear Redo.

### Timeline playback and scrubbing

The viewport plays the entire timeline by default. Its timecode, frame-step controls and skip-to-end
use movie time. Playback continues through each shot without selecting it in the inspector.
Clicking a clip selects it and moves the playhead into that clip; **Split at playhead** splits the
shot under the playhead. Keyframe attachment times remain relative to the selected shot and are
available only while the playhead is inside it.

Click or drag the **time ruler** above a clip to seek. Clicks in the blank area beyond the clips are
ignored. Drag the playhead's triangle or vertical line to scrub; dragging beyond the timeline clamps
to its start or end. Scrubbing pauses playback and resumes on release if it was already playing.
The ruler and playhead share the scrolled, zoomed timeline coordinates.

**Timeline · cuts** uses native playback with source trims, still images, title overlays and active
audio regions. It references existing movies in place, loads metadata asynchronously and limits
the preview canvas to 1280 pixels on its longest edge. Missing or unrendered shots keep their place
and show a placeholder, so later shots do not shift. Selecting a clip reuses the loaded timeline.
Transitions use a cut at the incoming shot's start while preserving the edited movie duration.

Source-only timelines with cuts, centered source audio and clip volumes up to 200% play directly
through AVFoundation, including accepted Swift LTX/H3 takes when Python is unavailable. Source
trims and per-clip volume remain applied. Separate audio regions, source pan, overlapping transitions
or larger volume settings still prepare the canonical soundtrack through the existing audio mixer;
these edits are never silently omitted to enable native playback.

**Render movie preview** builds a reduced-resolution movie with the actual transitions, titles,
audio crossfades and finishing mix. It is the more accurate check before export. Use **Timeline
playback** to return to immediate editing. Rendered previews are rebuilt after edits and require
all shots to have media. Interpolation/upscaling are applied in final export. Waveform editing is
not yet available.

## LoRAs and groups

Open **LoRAs & Groups…** in Assets, or **Add / Groups…** in either clip inspector or the Draw Things
image workspace. The library has one search and **All / Local folders / Draw Things** source filters.
**Compatible with selected model**
starts enabled; turn it off to review other models and files needing classification. Source and
model labels explain where each adapter can run. Only applied LoRAs appear in the inspectors.
Image-workspace changes apply to the image draft without changing the underlying movie clip.

In **Runtime Settings → LoRA model folders** (also **Folders…** in the library):

1. Use **Add folders…** to link one or more existing LoRA libraries. Files remain in place.
2. Enable/disable each folder, choose whether to include subfolders, and optionally specify a
   training-model fallback for files without metadata. Checkpoint declarations always take precedence.
3. Use **Refresh folders** after adding or replacing files. Opening the library also refreshes it.
   Scanning runs in the background, reads headers without loading model weights, and reuses cached
   inspections for unchanged files. Overlapping paths and file symlinks are deduplicated.

The default folder is `~/Library/Application Support/WeeTodd Studio/Models/LoRAs`; **Open default
folder** creates it when needed. There is no automatic file move, download, conversion or upload.
Removing a folder removes its discovery results; explicit individual imports and saved clip/group
links remain. Missing drives produce a notice and do not delete saved settings. Scans stop at
1,000 candidates or 20,000 directory entries; use narrower folders if the limit is reported.

**Link individual files / adapter options** retains manual SafeTensors imports, including H3 Turbo
options. Choose the **Trained model** before importing. Files with missing metadata offer **Set
trained model** when the compatibility filter is off. Filenames never establish the model.
Specialized or unreadable adapters remain visible under **Adapters needing setup**, with a reason;
IC-LoRAs still require the dedicated reference/control Model Setup route.

For Draw Things, use **Connections…** to configure the server, then **Refresh** beside its library
section. Local servers must advertise their installed LoRAs through Model Browsing. **Add to current
stack** uses the server's exact LoRA ID and checks the selected connection/model. A `.ckpt` store
is not a native SafeTensors adapter; pointing Studio at that folder does not convert it. To use an
ordinary local adapter with Draw Things, install it there and refresh its catalog. Native generation
can link compatible SafeTensors files wherever they are stored, including a shared model folder.

| Selected clip / group model | LoRAs offered |
| --- | --- |
| MiniMax H3 | H3 |
| LTX 2.3 | LTX 2.3 |
| LTX 2.5 | LTX 2.3 and LTX 2.5, including mixed groups |
| Draw Things | Catalog entries advertised for the selected connection and model |
| Movie / Still | None |

Use **New group**, or **Save current stack as group**, name it, and set each member's strength.
Groups can be edited or deleted. **Add to current stack** preserves existing LoRAs and rejects
duplicates atomically; **Replace current stack** explicitly replaces them. Draw Things image and
video group actions use the same Add/Replace choices. Each entry has an enable toggle, a slider,
and an exact numeric strength field from 0 to 2. Disabling retains the strength and saved assignment;
disabled entries are excluded from generation, including missing files or unavailable server LoRAs.
Remove individual entries or an entire applied group from the inspector. Duplicate file application
is rejected, including overlap between an individual entry and a group.

Groups are reusable templates stored beside Global assets. Application creates independent linked
Clip Assets and copies strengths, enabled states, adapter details and group labels into the clip. Editing/deleting the template
does not change existing clips, and clip strength edits do not change the template. Project saves,
autosave, undo/redo, duplication and splitting preserve applied settings. Movie and clip job exports
embed the flattened renderer stack, so headless execution does not need the group library.
Collect Media preserves shared LoRA file paths; it does not copy model weights.

Training versions identify candidates, not a guarantee of compatibility or quality. The shared
renderer still checks actual projection targets, dimensions, scaling and recipe restrictions before
weighted work. Specialized IC/control/reference adapters remain in their task recipes. Native H3
four-step Turbo adapters can be selected through the library as described below; other specialized
sampling recipes keep their existing contracts.
Switching a clip to an incompatible engine retains its settings and marks the clip as needing
attention until incompatible LoRAs are disabled or removed. Rebuild the app and use an updated managed renderer
source snapshot when upgrading; existing private runtimes retain their installed source.

Validation covers model filtering, mixed groups, independent clip strengths, serialization, duplicate
rejection, split asset ownership and movie/clip recipe export. The native app was exercised with
small synthetic header fixtures; these checks do not qualify LoRA visual quality or every adapter.

### Native H3 Fun ControlNet

With Swift H3 enabled, use the dedicated Fun ControlNet setup preset and attach one preprocessed
Canny, depth, HED, MLSD or pose movie as Control. The route uses the supported T2VA branch and
shared native video VAE/sampler and freezes the guide and checksum. It requires the supported
dense full-width T2VA checkpoint, no LoRAs or motion context, strength from 0 to 1, each output
side at most 2,048 and total canvas pixels at most 768 × 1,344. Task controls are validated. A direct 384 × 256, 73-frame test at 24 fps completed in 663.45 seconds with 32 kHz
stereo audio, four decoded previews and staged release. Peak Swift-worker footprint was 11.80 GB,
excluding FFmpeg. It used four Euler evaluations without a Turbo adapter; the result has not
received visual/control-quality approval. Installed Studio lifecycle and broader quality remain
unqualified. Follow the checkpoint's source terms in Model Setup.
A separate same-guide/checkpoint encoder comparison reduced MLX stage peak from 10.71 to
5.48 GB with byte-identical raw latents and zero residual allocations; encoding took 14.17
versus 11.13 seconds. This control-guide-only optimization does not establish a new
whole-generation speed or memory result; the full render above predates it.

### Native H3 Turbo

1. Choose a standard native H3 model and Text to video, Image to video, or First and last frames.
2. Open **LoRAs & Groups… → Link individual files / adapter options**, select **H3 Turbo (4 steps)** under Adapter, and import a compatible
   SafeTensors LoRA. For files without reliable profile metadata, this is an explicit declaration
   that the adapter is intended for four-step inference; its filename is not evidence.
3. Add it to the clip or a group and set its strength. **Steps** displays four actual transformer
   evaluations; the renderer stores five schedule points. Disabling Turbo restores the saved
   standard Steps override. The saved value is retained while Turbo controls the effective schedule.

**Adapter file details** exposes layout and the optional linked AdaLN input grid. Adapters with
AdaLN targets need that grid; the validator reports missing or incompatible auxiliary data before
weighted work. The experimental Swift H3 route accepts up to four distinct ordered compatible
ComfyUI-format adapters with independent strengths, including sparse projection sets and variable
ranks. Installed converted Lightx FL2V four-step adapters with baked scaling pass
header and projection checks. The full-rank adapter completed one signed-app audiovisual
render; resized-rank output and broader quality remain unqualified. Other LoRA layouts,
including AdaLN, remain on the existing route. Unsupported reference/control tasks,
conflicting acceleration recipes, explicitly incompatible step metadata and invalid layouts are
rejected. Existing custom recipes containing their own Turbo stack retain their saved schedule.

Header checks and regression tests verify parameter transport and compatibility checks. Installed
explicit-alpha and baked-scale Turbo adapters each have a real Swift render check; multi-adapter visual quality, timing,
memory fit, and motion-continuation quality remain unqualified.

### Clip continuity in Studio

Native H3, LTX 2.3 and LTX 2.5 clips have a **Clip Continuity** section:

- **Independent** preserves ordinary generation and is the default for existing clips.
- **Match previous frame** extracts the accepted source take's last visible frame, including
  trims and variable-frame-rate footage. It becomes the effective first image; stored first-frame
  attachments remain available when continuity is disabled. Last-frame guidance and LoRAs remain.
- **Continue scene** carries motion and sound using the selected model's supported route.
  H3 continues an accepted take using saved motion context; LTX 2.3 extends an accepted source video.
  Select the immediately previous clip or another earlier clip for these routes. LTX 2.5 connects
  neighboring shots and renders the complete group together, as described below. These routes are
  experimental; the inspector explains the required source and what will be generated.
- **Extend previous take** is LTX 2.5's separate source-video extension option. It generates only
  the new shot from an accepted take. A changed source take, trim or file invalidates preparation;
  late renders remain inactive takes.

For H3 motion continuation, enable **Save motion context** on the source, render it, and accept
that take. Saving is also inferred when a later H3 clip explicitly depends on it. The visible endpoint
must match the generated endpoint within half a frame; Prepare suggests compatible durations.
The target must use matching model components, dimensions, sampling and LoRA settings.

With **Use Swift MLX for H3** enabled, frame matching uses native AVFoundation extraction
and preserves the original editor attachments. Motion context currently requires text-only
H3 generation and matching settings; image/reference/A2V context saving is rejected.
Swift v2 `latents.f32` artifacts are distinct from Python v1 `latents.safetensors` artifacts.
Saving keeps the complete terminal 17-frame grid and makes the accepted clip use that
aligned duration, shown in preparation. Trimming its ending invalidates motion continuation.
The worker fully verifies manifest/payload hashes before loading weights.

LTX motion continuation needs a visible source with audio, at least 49 native frames long. Studio
prepares that bounded tail, preserves synchronized timing, and adds only the new segment to the
timeline. New durations must be multiples of eight native frames. LTX 2.3 requires compatible
Dev one-stage or original distilled weights; LTX 2.5 requires distilled two-stage generation.
With **Use Swift MLX for LTX 2.5** enabled (experimental), Studio prepares this 49-frame
tail in Swift and runs the native worker. **Extend previous take** uses a 25-frame tail
and supports the after direction in that worker. Both require embedded audio; the
selected source, dimensions, duration and recipe are checked before model loading.
Endpoint/audio/reference attachments and unsupported IC/MSR or single-stage combinations are
rejected without discarding settings. Use **Match previous frame** for endpoint-guided generation.
LTX already-distilled models do not need an H3 Turbo adapter.

### Continuity troubleshooting

| Symptom | Next action |
| --- | --- |
| A saved H3 shot says the scene requires LTX 2.5 | It still belongs to an LTX 2.5 group. Use **Separate this shot**, or switch it back to LTX 2.5. Updated model switching separates incompatible shots automatically and supports Undo. |
| H3 **Continue scene** needs saved context | Render and accept a compatible H3 source with **Save motion context** enabled. Keep its visible ending at the original endpoint. An LTX take cannot supply H3's internal motion context. |
| **Continue scene** renders every LTX 2.5 shot in the group | This is the grouped route. Review and accept the complete movie together. Choose **Extend previous take** when you want source-video extension of only the next shot, subject to its input restrictions. |
| An older LTX 2.5 scene flashes at joins | Regenerate with the updated renderer and **Automatic** boundary image guidance. The correction changes generation; opening an old take does not repair its pixels. Inspect the new take before replacing your edit. |

### Continuous LTX 2.5 scenes

Use **Continue scene** (experimental) for connected local LTX 2.5 shots that should play as
one continuous scene. The native engine carries original video and audio latents between
overlapping sampling windows. Scenes with a later first-frame image decode groups
between explicit image cuts separately and join at the exact editorial boundary;
otherwise the selected full or
bounded decode mode applies. Frame matching and
source-video extension remain separate choices for other workflows.

With **Use Swift MLX for LTX 2.5** enabled, the current Swift route accepts text scenes
with compatible ordinary LoRAs and one **First frame** image on any shot. Choose
**Image to video** for that shot. An **Audio to video** scene can use one consecutive
interval of the same source file on every shot, with an optional first-frame image on
each. Audio-driven shot durations must land on the eight-frame grid. The worker
preflights the whole group and produces grouped review ranges.
Studio uses one full video decode by default. With **Use Swift MLX for LTX 2.5** enabled,
choose **Video decode → Bounded decode** in the scene inspector to cap each decoder window
at 361 frames; headless recipes can set their own aligned cap. The choice is saved with
the scene leader, frozen into the prepared recipe, and shown during take review. Switching
to the Python renderer requires choosing **Full decode** first. One signed-worker
30-second scene rendered with no visible cut in inspected decoder-join
contact frames. Broader long-scene quality remains unqualified.
A StudioStore test using that real movie passed grouped review, acceptance and project
round-trip with a test bridge for preparation/rendering. A separate two-image,
continuous-audio Studio job then completed native preparation, rendering, grouped
acceptance, saving and reopening with Python unavailable. Select the Python renderer
for specialized controls or checkpoint-resume behavior. Those options remain available
in Studio.

Continuing windows reuse interior video history while regenerating the previous window's final
sampled video latent with future context. **Boundary image guidance → Automatic** applies every image
once, at its requested strength, in the first window covering its timestamp. Later windows inherit
its influence through motion history. Reapplying the image in that overlap can compete with the
history it already influenced and produce a dark pulse. **Strict** repeats image guidance in every
covering window for explicit control. All attachments, timestamps and requested strengths remain
unchanged, including the scene's first and last images. Preparation's **Boundary image routing**
shows the window using each boundary image and the windows inheriting it. These are generation
changes, with no added output crossfade. Overlap duration and native audio history remain intact.
VAE-encoded source-video extension retains its complete history; it does not reuse a sampled
window's terminal state. Regenerate an existing scene to apply the fix to its movie.

1. Set the first shot's connection to **Independent**.
2. Set each following shot to **Continue scene**. It joins its immediately preceding shot.
   When converting a frame-matched shot, choose **Keep current frame match** to save its current
   predecessor frame as an explicit first-image anchor, or **Use attached images** to use the
   shot's existing image attachments. The original assets remain available.
3. Use compatible LTX 2.5 components, sampling settings and ordered LoRA stacks for every member.
   Prompts, durations and seeds can differ. Keep the complete scene within two to six shots and
   30 seconds. Native timing resolves on eight-frame intervals; preflight reports resolved ranges.
4. Review the shared sound and music instructions in **Clip Continuity**. These belong to the first
   shot and appear when any scene member is selected.
5. Select any member and choose **Prepare scene** or **Generate scene**. Preparation covers the
   entire group and includes every member prompt, image anchor and resolved duration.
6. Review the complete generated movie, including sound and each join, then choose **Accept entire
   scene**. All member shots use ranges in the same movie; their earlier takes remain available.

First, last and timed images are mapped onto the full scene timeline, including the last visible
frame. Review these images when changing an existing project to a continuous scene. Frozen frame
matches become ordinary image anchors and remain stable when an older source take changes.
Preparation flags joins where different files guide consecutive end/start frames. Review those
pairs: competing full-strength images can cause a sudden change even with native motion context.
Automatic uses every guide once and avoids repeating it in incoming overlaps. Review boundary
images together; contradictory poses or lighting can still produce a sudden change.
Conflicting anchors and unsupported combinations fail before model loading. MSR/reference inputs,
control adapters, CFG++ and DFR are not qualified together with continuous scenes. Source audio
can accompany first, last and timed images: attach consecutive intervals of the same original song
to every member. Audio-driven scene durations must land on the eight-frame grid; other lengths
remain independent clips. Both generation stages freeze the source audio, and final publication
retains the original source interval rather than resynthesizing it.

The shared sound description and native audio context reduce independent audio restarts. A prompt
such as “no music” remains probabilistic. For exact soundtrack control, switch off **Use generated
scene sound in movie** and add an audio track. The scene review plays the generated audiovisual
movie; final movie playback/export also applies the timeline's source volume and audio tracks.

Changing any member's generation inputs invalidates preparation. A late result remains a review
artifact and cannot replace changed shots or another document. Scene versions restore together;
disconnect a shot before selecting an older individual take. One acceptance is one undo operation.

On the Python scene route, completed native sampling windows are retained as validated job-local checkpoints for retries.
Reusing the same prepared recipe can resume those windows after cancellation or decode failure;
changed input identities and corrupt checkpoints are rejected. Exported movie jobs generate a scene
once and apply its ranges to every member. Clip-only job export requires disconnecting the shot or
exporting the whole movie so the group is not silently split.
Each interactive retry writes a separate candidate movie while sharing the prepared recipe's
validated checkpoints. Acceptance waits until the active attempt finishes.

Timing limits apply after native-grid resolution. For example, 30 seconds at 25 fps rounds beyond
the current 30-second qualification limit; choose a supported shorter duration. Six five-second
shots at 24 fps resolve to 30 seconds exactly. Unit tests establish contracts and lifecycle behavior,
not seamless output or guaranteed sound instructions; inspect the generated scene before delivery.

With staged memory enabled, long convolutional-VAE scenes decode the assembled timeline in bounded
temporal tiles before decoding audio. This reduces peak decode memory at an additional time cost;
ordinary clip decoding and the diffusion VAE retain their existing behavior.
The first six-shot Rill qualification produced an exact 30-second movie with retained first/last
images and shared audio state. Some joins still showed framing/appearance changes, so the feature
remains experimental. See [implementation status](../STATUS.md) for measured timings and limits.

In the 2026-09-16 Studio comparison, LTX 2.5 Q8 distilled (8+3 evaluations) completed the same
Approach shot from an accepted H3 predecessor in 1:38 with frame matching + last-frame guidance,
and 1:59 with motion/audio continuation. The motion render contained 49 source and 120 new frames;
Studio correctly selected only the five new seconds. Both outputs had stereo 48 kHz audio and
decoded successfully. Scene appearance persisted, but the motion take crouched more than the
prompt requested. This checks the exercised source-tail route, not LTX-to-LTX chaining or a combined
reference + first/last frames + motion mode; that combination remains unsupported.

Automatic native model selection filters known task limitations, including text-only H3 encoders
and LTX 2.3 Single-pass 1.1. Non-Custom automatic presets prefer matching tasks and compatible
distilled LTX profiles. Explicit model selections remain authoritative.

Prepared media belongs to its job; exported jobs freeze accepted source takes and support safe
retry. Render and accept a queued predecessor before exporting its dependent clip. Collect Media
keeps H3 manifests and latent payloads together. These controls are native-engine features;
Draw Things retains its existing conditioning and generation controls.

### Experimental native H3 continuation

The shared headless renderer can save and reload a bounded synchronized video/audio latent tail.
Studio's motion continuity controls use this same opt-in contract. It remains available directly
to headless clients; the app does not maintain a second sampler. The JSON and file-content
identity rules below describe the Python version-1 path. Swift uses `"version": 2`,
checks component metadata identity and stores `latents.f32`; artifacts cannot cross engines.
The Swift text-only path keeps terminal alignment when saving, rather than rejecting the
aligned editorial-duration expansion. See [Clip continuity](#clip-continuity-in-studio).

Add this to an H3 headless recipe to save context from its completed take:

```json
"continuation": {
  "version": 1,
  "context_frames": 22,
  "save_context": true
}
```

The result reports a `.continuation/manifest.json` path and SHA-256 digest. A subsequent recipe
supplies both as `source_context` and `source_manifest_sha256`. Loading context makes the requested
duration describe **new visible footage**: publication removes the repeated video/audio prefix and
any extra alignment frames. Saving context again is rejected if the generated tail was trimmed,
because that tail would describe footage outside the visible take. The error offers eligible nearby
durations. With 22 context frames, a continuing take that saves its tail needs a multiple of 17 new
frames. Save-only first takes retain the ordinary H3 aligned output duration.

Only text-to-video and first/last-frame tasks are accepted initially. Context checks cover canvas,
timing, model components, adapters and sampling settings. Checksums detect changed manifests and
payloads before weighted work. Context export is bounded and atomic; existing contexts are not
overwritten. Components retain staged unloading. The first identity verification reads model files
to hash their contents. A bounded local SQLite cache reuses those checksums while each resolved
file's device, inode, size, modification time and change time remain unchanged. Unavailable or
corrupt caches fall back to full hashing; `WEETODD_DISABLE_MODEL_HASH_CACHE=1` forces re-verification.
Generated context manifests and payloads always receive complete checksum checks.
For first/last-frame motion continuation, Studio's resolved prompt replaces the standard leading
Picture alignment times with the actual sampled-window anchor times, including reused context.
The shot direction stays intact and the exact resolved prompt remains available before generation.

The 2026-09-16 Studio exercise completed a six-shot, 30-second H3 movie with Q8 FL2VA, first/last
frames, LightX2V Turbo at strength 1, four evaluations and 22-frame motion context. Export verified
720 frames at 24 fps with stereo 48 kHz audio and no decode errors. Scene/action continuity was
recognizable; some mid-shot exposure variation remained. This is scoped visual qualification,
not a guarantee of seamless results, reference-plus-endpoint support or fit on lower-memory Macs.

On the tested 256 GiB M3 Ultra, switching the first shot from lower-memory to larger-workspace
paging reduced sampling from 7:23 to 3:15 (whole job 8:59 to 4:06). Outputs were similar rather
than identical: whole-video SSIM 0.9741. Continuing shots took 5:19–5:58. Whole-job comparison also
includes the benefit of cached model checksums. Motion continuity stays opt-in pending broader
scene, audio and memory qualification; these results do not establish performance parity with
Draw Things.

## Status and actions

| Clip color | Meaning |
| --- | --- |
| Green | Generated source matches the current generation settings and linked inputs. |
| Yellow | A generated clip's settings or inputs changed, or a different version was selected. |
| Orange | A new clip has the required basic setup and is ready for full preflight. |
| Red | A generated clip has missing configuration or its last preflight needs attention. |
| Blue | Imported movie/still/sequence clip. Missing files appear in Actions. |

The status area's **Actions** button orders blockers before generation work, then save reminders
and suggestions. Each item opens the relevant clip, prompt, runtime settings, save or job export.
Output settings do not unnecessarily invalidate the source generation; finishing happens during export.

## Movie settings and finishing

Movie dimensions, frame rate, interpolation/upscaling, fit and format are inherited by every clip,
with optional clip overrides. Each clip finishes separately, then assembly conforms it to the movie
canvas and frame rate. Supported outputs are H.264 MP4/MOV, ProRes 422 HQ, or a PNG sequence plus WAV.
Current intermediate clips use H.264; ProRes and PNG output are not end-to-end lossless masters.

Finishing order is resize/upscale, then interpolate, then transitions/titles/audio assembly.
The assembled video is conformed again to the movie frame rate so AAC packet padding and
transition timestamps cannot leave gaps in its frame cadence. Final validation checks the
exported streams and timing before reporting success. Set the finished length through clip trims
and transition overlaps; model frame-count rounding can make generated clips longer than requested.
RIFE supports integer 2×/3×/4× interpolation with its configured MLX executable and weights.
MetalFX spatial upscaling uses the native helper. Neither method downloads models silently.

Experimental MetalFX frame interpolation supports 2× and requires genuine per-frame guides at the
processed clip's resolution and timing. Set depth and motion folders in the clip's advanced settings:

- `000001.f32` through the last input-frame index: tightly packed little-endian float32.
- Depth is one channel per pixel; motion is two channels of backward displacement in pixels.
- The depth folder must contain `camera.json` with `nearPlane`, `farPlane`, `fieldOfView` in degrees,
  and Boolean `depthReversed`, describing the actual guide camera.
- Guides must already match trimming, fit, dimensions, and frame rate. The app does not synthesize
  missing depth/motion or imply that ordinary movie files contain these buffers.

## Headless movie and clip jobs

Use **Movie → Export Movie Headless Job** or **Export Clip Headless Job**. The job retains
local media/model paths; collect the project first if media needs to travel. Keep its prepared-input
folder beside the job, and retain the recorded workers, models and FFmpeg installation.

When every clip uses the selected native Swift H3/LTX route or an accepted movie take, Studio exports
`weetodd-studio-native-job-v1`. It freezes the original edit, exact prepared `weetodd-headless-v2`
recipes, preparation reports and input observations. Export and execution use the same native
preparation and workers as Studio, with no Python prerequisite or inference fallback. This host
supports cut movies with embedded source audio, trims, per-clip gain from 0–200%, fit/fill, output
frame rate, and MP4, MOV or ProRes finishing through FFmpeg. It explicitly rejects titles, added
audio, source pan, transitions, PNG sequences, upscaling/interpolation, differing per-clip finishing
overrides and Motion Fidelity rather
than omitting those settings. Native continuous-scene export retains the complete scene group and
its prepared member ranges, including full and windowed publication; export the whole scene rather
than one member. A dependent continuity clip needs its source take rendered and accepted before export.
Pending Ripple generation is rejected by this exporter. Generate, review and apply the Ripple take
in Studio first; its accepted movie source can then be exported without replacing the edit request.

Run the app's CLI with the editor closed:

```bash
"/Applications/WeeTodd Studio.app/Contents/MacOS/WeeToddCLI" \
  --job Movie.weetodd-job.json --output-directory Render --preflight-only
"/Applications/WeeTodd Studio.app/Contents/MacOS/WeeToddCLI" \
  --job Movie.weetodd-job.json --output-directory Render --resume
```

If the app moved, add `--h3-swift-worker /path/to/WeeToddH3MLXWorker` and/or
`--ltx25-swift-worker /path/to/WeeToddLTXWorker`. All pending recipes and reused source intervals
preflight before the first generation. FFmpeg encoder, filter and muxer availability is also
checked before worker admission; unsupported CLI options are rejected rather than ignored. Workers run sequentially and retain decoded previews,
progress and release events in full JSONL logs. The host accepts takes into `result.weetodd`,
preserving scene identity and continuation metadata, then assembles the movie. Cancellation passes
to the active worker or FFmpeg process. Resume verifies the frozen job, selected worker hashes,
source observations, completed takes and finished movie; changed jobs or inputs require a new
export/output directory. Each output folder is locked against concurrent execution. The model's
minimum working memory still applies. Result receipts retain `generations` as the total job
count and distinguish `newlyGenerated` from `resumedGenerations`; a completed resume reports
zero newly generated takes.

Qualification on 2026-10-02: deterministic worker/AV fixtures passed byte preservation, admission
before inference, cancellation, finishing, acceptance, reopening and resume checks. A saved MSR
Studio job exported through native preparation and passed the actual CLI's installed-worker preflight
with Python unavailable. The same immutable exported job subsequently completed real Swift
MSR inference, delivered two decoded previews and complete stage-release events, retained its
accepted project and assembled the movie without Python. The accepted project reopened and
verified resume reported zero newly generated takes and one reused generation. Both full worker
logs and the take hash remained unchanged. That real run predates optional take-statistics and Clip Assets
metadata enrichment; the final metadata path has separate audiovisual fixture checks. This does
not establish all-task exported-job qualification. A corrected CrossView/Ingredients Studio
job also completed native CLI inference with all video/audio/muxed media byte-identical to
Studio, then resumed with zero new and one reused generation.

Other backend selections retain the existing `weetodd-studio-job-v1`–`v4` Python-hosted export and
companion instructions. Those jobs support their existing titles, transitions, added audio and
configured enhancement finishing. Their CLI finds the recorded Python runtime automatically;
selected Swift H3/LTX clips still infer in the recorded Swift workers. A previously qualified
five-second LTX 2.5 T2V export used that Python host, rendered in Swift, assembled a 1920×1080 stereo
movie and reused its take on resume. This is separate evidence from the new native host.
Developers can run **legacy jobs only** through the existing Python client:

```bash
.venv/bin/python scripts/render_headless.py \
  --job Movie.weetodd-job.json --output-directory Render --resume
```

Legacy clip jobs retain intersecting titles/audio shifted into clip-local time. Native clip jobs
reject added timeline titles/audio and preserve the selected clip's source interval and settings.
Swift H3/LTX workers share the cross-process inference lock with Python/MLX jobs. Existing
composable ComfyUI nodes retain their Python samplers; recipe-backed Swift MLX nodes use these workers.

## Validation and next release work

Run `python scripts/validate_project.py --profile studio` for Swift tests and the Python Studio
bridge/packaging tests. Add `--profile workflows` or `--profile remote` for those integrations.
Release packaging can target a separate bundle while the development app stays open:

```bash
python scripts/build_studio_app.py --configuration release --output /tmp/WeeTodd-Review.app
```

Separate output preserves the default bundle and saved signing identity. The selected output is
still protected against replacing a running app; all nested tools and the completed bundle are
signed and verified.

Opening or creating a project starts a separate undo history. Before leaving an unsaved document,
Studio writes a per-session project snapshot and its source-file metadata under the application
data directory's `Recovery` folder. A failed recovery write keeps the current document open.
These snapshots supplement the working-copy autosave; recovery history browsing remains future work.
Open/New also replaces the active startup snapshot immediately, so a restart restores that movie.
Render preparation belongs to its original document and clip. A render that finishes after clip
edits is retained as a version without replacing those edits; if its document was closed or its
clip deleted, Studio reports the saved output path instead of attaching it elsewhere.
New render versions preserve their usable source interval, including extension offsets. Switching
versions preserves relative trims where that interval permits; selecting the active version keeps
its existing trim, including older projects without interval metadata.
When an older appended extension has no recorded context boundary, switching to a newer version
starts at that version's known usable segment; historical trim offsets cannot be reconstructed.
Generate reuses a still-valid reviewed preparation and prepares again only when it is missing or stale.

Attachment digests use streamed background reads and are cached by file revision. Preparation
awaits the digest, and Draw Things generation verifies the file contents again before submission.
Image thumbnails decode off the UI thread to a bounded pixel size and share a 64 MiB decoded-image
cache. Movie preview's duration and skip-to-end use the complete movie duration.

The initial validation includes Swift document tests; real FFmpeg movie/transition/title/audio and
sequence/anchor tests; job locking, integrity and resume tests; a real LTX 2.5 generated job and resume;
a fresh private-runtime installation; and real MetalFX spatial/RIFE and guided MetalFX interpolation.
The private-runtime test produced byte-identical generated and assembled MP4s to the development
runtime for the one-second LTX 2.5 fixture. This is a narrow integration check, not universal parity.

Seed Undo/Redo was also checked in the release app with a focused field, after Tab, through the Edit
menu, with multi-digit replacement, and across separate editing sessions. Restoring the generated
seed restores the clip's green status without rerendering.

Before a broad consumer release: expanded model download coverage and tool packaging, sign and
notarize the app, test clean Macs and lower-memory hardware, qualify more conditioning combinations,
and improve live timeline playback. Useful next features are audio waveforms, proxy/cache management,
crash-recovery history, and a render-cost/memory estimate before queuing large movies.

## Motion Fidelity (De-Roping) · experimental H3

Select an H3 clip and open **Motion Fidelity** in the clip inspector. Enable **Use enhanced
motion**, then choose **Analyze** to inspect the expansion plan or **Enhance** to render it.
The original movie remains in Versions and Clip Assets. The enhancement is a separate Clip Asset;
turn the option off to compare the original at the same playhead position. Changing enhancement
settings marks the clip yellow without invalidating its base generation. Actions links to pending
motion work. A stale or missing enhancement blocks direct movie export rather than silently using
the original. Export a headless job to process pending enhancements with Studio closed.

Choose **Edit Repair Prompt** to open the complete resolved repair prompt in a full-window editor.
Save keeps the nonblank text exactly as written, including surrounding whitespace; it does not
rewrite or reformat the prompt. **Use Recipe Prompt** clears the clip override and returns repair to
the selected recipe's original prompt. Changing or clearing the override makes only the existing
enhancement stale. It does not invalidate the base generation.

- **Adaptive** uses third temporal differences of H3 video latents, adjusted for the VAE's five-phase
  cadence, to allocate integer frame holds. It is a heuristic, not a reliable artifact detector.
  A quiet plan bypasses refinement. **Uniform** expands every source frame equally.
  Inspect the analysis before rendering: adaptive coverage can be sparse even during continuous
  action. Use Uniform when you want the entire clip treated.
- **Maximum hold** is 2–4×. Sensitivity affects adaptive coverage; refinement strength controls
  partial denoising. A value of 0.5 starts video at 50% noise; audio uses its corresponding
  shifted clock. The inspector displays strength numerically with 0.01 increments.
  By default, evaluation count is the ceiling of the recipe's full evaluations × strength.
  Enable **Set refinement evaluations** to choose 1–64 evaluations independently of strength;
  the initial suggested value is 14. This makes equal-budget strength comparisons possible
  without changing the base generation recipe. More evaluations cost time and do not guarantee
  better visual results. Existing projects retain their automatic evaluation counts.
  Long refinements report completed/total evaluations in the status area.
  The partial interval is resampled instead of taking the tail of a heavily shifted schedule.
  The seed belongs to enhancement independently of the base clip.
- Use an H3 T2VA repair recipe with at least 16 schedule points. The default uses the selected
  base render's recipe. Choose an explicit compatible repair recipe when the original used image,
  reference or audio conditioning. Its prompt and components govern refinement. A selected repair
  recipe may include standard LoRAs active for the full schedule; they apply only to refinement and
  their application is recorded in the result. Turbo/distilled or staged LoRAs, FastH3/VDN, cache
  accelerators and extra conditioning are rejected before weighted work.
  Check the adapter's fused attention layout separately from its tensor shapes. For a
  ComfyUI-exported adapter whose Q, K and V rows are contiguous, set the recipe adapter's
  `qkv_layout` to `contiguous_qkv`. The native engine uses per-head interleaved rows; choosing
  `native_interleaved` for contiguous weights applies the wrong attention deltas even when every
  target shape passes validation. `auto` uses declared layout metadata when present, otherwise
  it currently defaults standard adapters to the native layout. An undeclared export layout must
  be established before rendering. The result records `qkv_permuted_targets` alongside the
  applied target count. Adapter strength and refinement noise strength are separate controls.
- Input is constant 24 fps, with frame-aligned trims, 32-pixel-grid dimensions and 60–345 source
  frames. Expansion is padded to H3's `17k+5` geometry and must fit the configured budget, at most
  345 frames. The current RGB conversion also limits expanded width × height × frames to
  160 million pixels. Split longer clips or reduce the budget; no automatic windowing is claimed.
- Expanded conditioning audio is stretched with pitch-preserving FFmpeg filters. Final output
  uses the original trimmed soundtrack, re-encoded to AAC, at the original duration and frame
  rate. Silent sources receive silence. Model refinement can still alter mouth motion or identity.

Movie/clip jobs with Motion Fidelity use `weetodd-studio-job-v2` and embed their repair recipes,
including a clip's resolved prompt override. The bridge's read-only `motion-prepare` action returns
the resolved prompt, original recipe prompt and whether the clip uses an override; it validates the
recipe without writing preparation files or loading model weights. Whitespace-only or non-string
overrides are rejected before enhancement starts.
For these legacy Python-hosted jobs, `WeeToddCLI` runs generation, enhancement,
upscaling/interpolation and assembly serially.
Completed enhancements are hash-checked on resume; an interrupted enhancement restarts that clip's
refinement. It does not resume inside transformer sampling. The app-managed native runtime needs
this renderer revision; install a fresh runtime when adopting it and export a fresh job. Jobs with
the option off retain v1 behavior. Project files and Collect Media retain both source and enhancement,
including the enhancement's analysis and repair-recipe record. Model weights remain shared.

ComfyUI exposes **H3 Motion Fidelity Settings** and **H3 Motion Fidelity Refine**. Connect H3
Components, Generation Config and Motion Settings; provide a source movie path, native trim and
repair prompt. Analyze-only defaults on. The adapter runs the same isolated helper used by Studio
and jobs, returning a movie path and JSON analysis report. No paid workflow or external node pack
is required. The settings node's optional **evaluations** input uses 0 for the existing automatic
behavior, or 1–64 for a fixed count. Headless motion settings use `"evaluations": 14` for a fixed
count; omission or `null` keeps the automatic behavior. A change invalidates enhancement only,
while the original generation remains reusable. Existing workflow contracts remain compatible.

Validation includes a real native MLX partial-denoise render, exact 73-frame recovery at 24 fps,
32 kHz source-audio remuxing, mixed audio holds, optional bypass, old project decoding and separate
base/enhancement invalidation. Low-resolution fixtures establish execution and timing only;
they are unsuitable for judging motion fidelity. Refinement uses an explicit source-noise fraction
to avoid injecting near-pure noise from H3's heavily shifted generation schedule.
The motion-quality reference is a dense H3 boxing clip at 896×512, 124 frames and 24 fps, generated
with 20 schedule points (19 evaluations) and no FastVideo/VDN acceleration. Uniform 2× expansion
with strength 0.5 completed ten refinement evaluations over 260 padded frames, then recovered
exactly 124 frames with less than 1 ms AV drift. Matching-frame review retained the subject and
action, but changes were subtle and fast-glove blur remained. This does not establish a general
visual improvement or qualify dialogue/identity preservation across scenes.
The packaged CLI completed a v2 enhanced movie and reused the same final hash on resume. Native
Studio Enhance added a separate Clip Asset; toggling the source comparison reused it. The complete
suite passed 1,349 Python tests (one skip) and 13 Swift tests for this checkpoint.
LTX, imported-movie UI support, regional editing, overlapping long-clip windows, side-by-side viewing,
per-stage latent resume and broad dialogue/identity qualification remain future work.

The repair-prompt and standard-adapter checkpoint passed 1,470 Python tests (two optional skips)
and 14 Swift tests. The packaged app's full-window repair editor was checked interactively in
Light and Dark modes, including exact multiline Save, Escape cancellation, recipe-prompt reset
and missing-recipe recovery. These checks establish implementation behavior, not a quality preset.
Matched tests must also verify the adapter export layout; a shape-compatible wrong-layout run is
not valid evidence for or against that adapter.

A native LTX 2.5 implementation is a planned follow-on. It can reuse the clip editor, enhancement
versions and headless job behavior, but needs a separate engine plan for LTX's `8k+1` frame grid,
conditioning clock, refinement schedule and bounded overlapping windows. The H3 motion adapter
cannot be applied to LTX. Existing LTX source-latent refinement and frozen audio provide building
blocks; temporal DFR remains experimental and is not equivalent to this expansion/recovery method.
LTX support must preserve source timing and pass identity, action, audio and seam comparisons before
it becomes an available Motion Fidelity clip option.

## Development files and cleanup

- `studio/.build/` contains Swift build products and the local app. Rebuilding refreshes the bundled
  renderer; an already installed private runtime retains its own source snapshot. Set up a new runtime
  when adopting renderer changes, and export new headless jobs for that runtime.
- Studio projects, collected media folders, headless job JSON and companion instructions are ignored
  throughout the repository because they contain user content and local file paths.
- Application Support contains user work as well as caches: generated clip versions, autosave,
  global assets, imported recipes, jobs and runtime receipts. Back it up before manual maintenance.
  Collect Media is the supported way to preserve a movie's referenced media for portability.
- Prior managed runtimes and render directories are retained for existing jobs; there is no
  automatic cleanup or rollback selector for them. Do not remove a directory still referenced by
  a project/job. Disposable decoded-audio and preview-mix caches have automatic bounded eviction;
  saved driver/export artifacts remain durable.

Run `swift test --package-path studio` and
`python -m pytest -q tests/test_studio_bridge.py tests/test_studio_packaging.py tests/test_studio_lora.py` before packaging.
The packaging tests exercise a source tree without `.agents/`, stale-bundle replacement, and failure
preservation without downloading Python or installing models.

## Draw Things — experimental

Draw Things is a central image/video inference route in WeeTodd Studio, alongside native engines
and imported movies. Its helper is an optional build component so native-only installations remain
possible. The shared Python adapter invokes that separately built Swift gRPC helper; it does not
import ComfyUI or load native MLX generation weights. Native projects and v1/v2 headless jobs remain
readable. Connection capabilities determine which models/tasks are available; native feature
support must not be inferred from a model appearing in the remote catalog.

### Build the optional connection runtime

```bash
python3 scripts/build_drawthings_client.py
python3 scripts/build_studio_app.py --configuration release \
  --drawthings-distribution studio/.build/drawthings
open "studio/.build/WeeTodd Studio.app"
```

The helper uses Draw Things' official `_MediaGenerationKit` product, pinned to community revision
`08e798b5ad59c3db78b2be53f0ed60b071653302`, and requires Swift 6 on Apple Silicon.
This revision adds H3 transport/configuration support beyond the older public wrapper release.
Building it downloads software dependencies, not model weights. The distribution includes the helper,
its hash manifest, licenses within a complete dependency-source archive, and instructions for
rebuilding with modified libraries. Studio lets users import a replacement executable. The synthetic
fixture server is a development test target and is never bundled with Studio.

The current community revision is GPLv3. The older public `media-generation-kit` wrapper's
[LGPL grant](https://github.com/drawthingsai/media-generation-kit#license) applies to that package's
distribution; it does not establish a grant for this newer direct dependency. The build commands
above are for local development. Publishing a bundled Studio/helper app remains pending resolution
of the GPL distribution requirements or an applicable upstream alternative license. The helper's
notices include this distinction and its corresponding source; WeeTodd's own source remains
Apache-2.0.

When upgrading an existing installation, Studio automatically uses its bundled helper if the saved
helper setting is missing or empty. An explicitly imported helper path stays selected, and existing
Python, model-recipe, and finishing-tool settings are preserved.

A normal Studio build can omit this optional distribution. Existing native generation still works;
Draw Things Connections then requires importing a helper executable. Python and FFmpeg remain
necessary for the shared job/finishing bridge.

### Connections, allowance, and CU

Open **Movie → Draw Things Connections** and save a connection:

- **Self-hosted gRPC:** enter the Draw Things server host/port, TLS choice, and optional shared
  secret. Confirm that this server has cloud offload disabled. A local-looking address alone does
  not establish that generation is self-hosted. The server must stay running.
- **Draw Things Cloud API:** create an API key in the Draw Things dashboard and enter it in Studio.
  The helper uses fixed official HTTPS/gRPC endpoints with TLS verification. It first obtains an
  authentication session and reads billing/free-request status; it does not change billing settings.
  Preparation requires explicit PAYG-disabled status, remaining free requests, and a fresh monthly
  record. The saved API key is checked even when discovery omits CU thresholds. Missing or invalid
  billing/quota fields block generation.
- **DT+ App Bridge:** discovery can be configured, but free-only generation is unavailable. The
  current bridge protocol does not expose a verifiable account/allowance/no-paid-fallback policy.
  Draw Things being open or subscribed to Plus is not sufficient evidence.

**CU measures the estimated work of one generation.** It is separate from the remaining monthly
request count and is not a currency balance. Studio shows the estimate for the resolved settings.
When the service advertises CU thresholds, cloud preflight uses the lower threshold until generation
authorization confirms account class. A request equal to or above that limit is refused. If neither
Echo nor Hours publishes thresholds, Studio shows **CU limit checked by Draw Things on submission**;
it does not invent a numerical limit. A verified remaining free allowance with PAYG disabled permits
requesting server authorization when Generate is pressed. A fresh free-only authorization is still
required before the generation RPC; paid, Boost, unknown, and expired grants are rejected. Reduce
dimensions, duration, or steps if the server refuses the job. Allowance is rechecked at generation time.

This release supports `freeOnly`; it neither selects PAYG/Boost nor silently falls back to them.
Generation authorization is performed only after local files, model availability, connection, and
output creation pass. An interrupted authorization/submission may already have consumed a request;
there is no automatic retry. A live Studio LTX 2.3 Cloud API run verified the saved key and free
allowance, completed generation, and saved 81 frames at 768×448/25 FPS with 48 kHz stereo audio.
This validates that route for the tested request, not every cloud model or account configuration.

Credentials stay in Keychain, or in an explicitly selected runtime environment variable for CLI and
ComfyUI use. They are never embedded in project requests or job JSON. An exported credential reference
identifies how to supply a secret; it does not include that secret. The Python runner accepts
`WEETODD_DT_CREDENTIAL` or a profile `credentialRef` of `env:VARIABLE_NAME`. Treat exported project
prompts and media paths as private even though credentials are excluded.

### Local workflows

Open **Director** in the toolbar, or **Movie → Workflows…**, to run the built-in movie planner or
import a user workflow. Director is the name of the planning assistant; versioned workflow files
remain its execution format. New guided plans organize review around **Brief**, **Subjects** and
**Shots**. Choose detailed review before starting for separate intermediate approvals. Saved jobs
retain the exact approval requirements in their saved definitions.

Director saves intake, selected step and unfinished typed review edits separately from approved
outputs, under its local `Director` data folder. Closing and reopening restores these drafts.
**Save** and **Approve** include reference additions, removals and ordering without a model call.
An invalid imported job leaves the current session intact. After a failed or paused run, completed
outputs remain available. Applying a result checks the originating document session and inputs;
reopening a different movie cannot silently redirect a late result.
For a failed shot without an editable draft, **Retry with direction** lets you supply a specific
correction while keeping completed drafts and approved story choices. Once a shot has a draft,
**Edit… → Visible characters** lets you select its on-screen cast directly alongside action,
starting state, ending state and location. These saved edits do not call the assistant.

Before loading weights, the helper checks the actual text/image token budget: at most 24,000 UTF-8
input bytes, 4,096 input tokens including image expansion, and 1,024 output tokens. A task that
cannot fit stops with guidance to split it; source text remains saved. Large directly relevant
context is not silently clipped. Repeated interruptions no longer make an unfinished coverage
review appear ready.

Human review actions retain revision-linked changes and model-turn references in the atomic run
checkpoint. They are local provenance, **not permission to train on or export your work**. The
checkpoint remains bounded; when its storage limit is reached, the previous checkpoint is preserved.

**Execution history…** opens a read-only step inspector, including while a workflow is running.
Select a step for its purpose, recorded elapsed time, saved response count (including child
records), and chronological execution/reuse entries. New activity records distinguish actual model
requests, reused responses, step-level retries, cancellations and user-requested reviews. Expand
**Call and retry details** for timings, task/system-instruction excerpts, errors and request-text
fingerprints. A returned model response has not necessarily passed output validation; validation
failures appear separately. Timings include loading and validation; model-call times are nested
within step times, and human approval waiting is excluded. The panel reads the checkpoint off the
UI thread every two seconds; live activity elapsed time updates once per second.

**Model turns…** inside a selected step opens every archived model request or reused response,
with full **System**, **Input**, **Response**, **Images** and **Details** tabs and a copy button.
Messages load one turn at a time from a read-only SQLite connection; the list loads in pages of
100 and refreshes during execution. Details include model/runtime identity, decoding settings,
request fingerprint and recorded JSON/schema validation results. A schema pass is not a semantic
or human approval. Errors, cancelled calls and malformed responses remain inspectable after retries.
Older checkpoints expose retained full messages and child responses; missing fields are explicitly
unavailable, and their checkpoint order is not claimed to be chronological.

New full transcripts are stored in `model-turns.sqlite` beside `run.json`, independently of the
abbreviated overview. No turns are silently rotated out of this database. It is capped at 16 MiB
or a smaller share of the workflow's declared disk budget, with room reserved for checkpoint and
SQLite journal writes. New model calls require at least five artifact slots. If the archive fills,
the workflow stops explicitly before admitting another request; keep the run folder and start a
new run. Image files and model weights are never copied into the log. These local records include
your prompts, responses and referenced paths; inspect them before sharing.

History is bounded to the last 40 activities and 20 detail entries per activity. Older omitted
entries are disclosed, and counters cover the whole retained activity even when details are trimmed.
Older runs remain inspectable but only expose their saved timing/responses; missing past retries
and regenerations are not reconstructed. The history does not modify approvals or trigger generation.

Set movie length, preferred clip length, FPS, idea and optional reference images, then use
**Run next** to inspect each step or **Run remaining** to continue. Choose a step to inspect
its output; **Regenerate selected** invalidates its dependent results. **Pause**, **Resume last**
and **Open job…** preserve completed work. **Export job…** writes a local job executable with
`python scripts/run_studio_workflow.py /path/to/job.json`, using referenced files in place.

The Prompt Assistant’s **Step-by-step…** button opens staged image observation, edit planning,
sequential editing and review. It returns a proposal for explicit application. Built-ins require
Qwen3.5 4B; custom text-only definitions can use 9B. Model-generated reviews may be incorrect.
You can paste a detailed `[Shot 1]` … script into Movie idea. Matching shot counts preserve
source order; timestamps are checked against the movie settings. Model output formatting errors
are distinguished from input-length errors, and one missing outer JSON closer can be recovered
without changing its values. Your source text remains saved unchanged.

Short authored shots that fit the 300-character beat contract are copied directly in guided
planning, with validated leading timing metadata removed from the action. Longer shots use a
bounded summary request. Recognized source speech is copied without normalizing its punctuation;
invented, changed or duplicated quoted words are rejected before review. Approved clarification
answers are part of the permitted source. If their dialogue has no safe shot assignment, Director
stops with guidance instead of silently omitting it. This is a bounded parser for quoted speech,
speaker labels and H3 dialogue tags, not a universal screenplay or language parser.

The separate legacy **Movie planning** workflow observes references once, writes a structured story, then pauses for
**Approve step**. After approval it plans one small action with start/end states per clip and pauses
again. Select **Clips** to edit states, approve individual clips, or **Repair…** a selected clip with a specific correction.
Approved choices must be unlocked before editing or repair. Changes revalidate dependent clips;
unaffected approved clips are reused. **Run remaining** builds endpoint descriptions directly
from approved states and checks timing/links. Structure validity does not certify creative quality.
Older imported v1 definitions remain supported. Planning workflows do not generate endpoint
images/video or insert timeline clips automatically. After importing and reviewing a plan, use
**Shot List → Add approved selection to timeline** to create clips explicitly.
Workflow task-LoRA/QLoRA loading and training are not available in the app. A separate small
text-adapter experiment has demonstrated training/save/reload, but vision compatibility and
quality gates remain unresolved. It does not qualify a user-facing training feature.

See the [workflow authoring and execution guide](../examples/studio-workflows/README.md) for
schemas, job bindings, storage limits, cancellation and the remaining milestones.
The [production evaluation guide](../examples/director-production-evaluation/README.md) describes
the frozen movie-planning regression cases, explicit local-model execution and measurement limits.

### Local Qwen3.5 Prompt Assistant

Open **Set up assistant…** in Director or the Prompt Assistant. You can reuse an installed model,
locate one on another drive, or download the supported Qwen3.5 4B checkpoint without installing
the Draw Things app. The managed renderer and separately linked local helper are still required.

The download is approximately 4.89 GB. Choose Studio's default model folder or an external folder;
**Download / Resume** continues an interrupted download in that location and verifies the pinned
SHA-256 before publishing it. Existing files are preserved. **Check text + image inference** makes
two small local test calls; it is explicit, separate from downloading, and does not certify creative
quality. A reused split checkpoint needs this check before selection in the setup sheet.

Studio discovers these dedicated text-generation checkpoints in the standard Draw Things and
Studio-managed model folders:

- `qwen_3.5_4b_i8x.ckpt` — Qwen3.5 4B, with text generation and vision.
- `qwen_3.5_9b_i5x.ckpt` — Qwen3.5 9B, currently text-only in Studio.

For another model location, use **Locate…** and select the installed checkpoint. Keep any matching
`-tensordata` sidecar beside it. Studio opens the files in place and does not convert or copy them.
Other Qwen checkpoints, including H3's truncated Qwen3-VL encoder, are not interchangeable.
**Rescan** checks the standard model folders. An existing model in an external/custom store
must be selected with **Locate…**; the absence of a scan result does not mean it needs downloading.
Studio remembers the selected checkpoint path across restarts.

For headless setup, use the configured project Python environment:

```bash
python scripts/setup_assistant_model.py catalog
python scripts/setup_assistant_model.py download --destination /your/model/folder
python scripts/setup_assistant_model.py health /your/model/folder/qwen_3.5_4b_i8x.ckpt --helper /your/WeeToddDrawThings
```

The [assistant evaluation probes](../examples/assistant-evaluation/README.md) measure exact content,
format and latency on synthetic tasks. They are separate from full-movie and hardware qualification.

Draw Things image and video requests accept seed `-1` for a fresh random seed at execution.
`0` and positive integers through `4294967295` are fixed seeds. Config imports and headless jobs
preserve `-1`; the shared adapter resolves it once per run before estimation and reuses that
exact value for generation and result provenance. A separate Check Settings & CU is a preview;
a later generation gets its own seed. The image editor offers **Random each generation**
(the default for new drafts) and **Fixed seed**. Existing fixed seeds remain fixed until changed.
**New fixed seed** chooses a number once; repeated generation then intentionally reuses it.

Enter instructions to draft new text or improve the current prompt, choose the output limit, then
**Generate text**. Review/edit the proposal and explicitly **Apply to prompt** or **Copy** it.
Applying changes only prompt text; it does not launch image/video generation or alter settings,
audio fields, references or LoRAs. Clip changes participate in Studio Undo. If the destination
or its prompt changed during the request, copy the result and reopen the assistant.

For visual assistance, choose the 4B model. The assistant displays labeled thumbnails of the
enabled image canvas/mood-board references, or the selected clip's attached images, including
first, last and keyframe roles. Check/uncheck images individually; **Load images…** adds other
local images without changing generation conditioning. Up to eight images can be included, in
the displayed order. The first eight candidates start checked. **Exclude all** selects text-only
operation. Each image is limited to 64 MiB, EXIF orientation is honored, and analysis uses a
whole-image RGB view up to 512 pixels per side with dimensions rounded to 32-pixel multiples.
The original files are unchanged. Smaller analysis images can miss fine text and tiny details.
Movie files/audio are not sent to the VLM; clip image attachments are still images.

Inference runs locally through the bundled Draw Things helper. Neither Draw Things'
gRPC server nor DT+/API credentials are required. The input limit is 4,096 tokens including
image tokens (also at most 24 KB of prompt/instruction text); output is selectable up to 1,024
tokens, with a warning when the limit is reached. Progress shows model loading, image processing
and output-token counts. Completion reports the number of images used and
elapsed time. Single prompt actions, Character Director and workflow jobs use a job-owned
Qwen3.5 4B session: consecutive calls reuse decoder/vision weights and up to four content-hashed
image sets (128 MiB combined
prepared tensors and encoded features). Attention KV, convolution and recurrent state are fresh
for every request. The session closes and unloads before review, on cancellation/failure, or when
switching models. Progress reports preparation, reading, writing and final unload. This is serial
resource reuse; simultaneous GPU batches are not enabled. The cache cap excludes model weights
and is not a total-memory limit. A single prompt action closes its session after that action.
9B text remains one-shot. Older helpers can use one-shot compatibility only after an explicit
startup rejection, before any generation is submitted.
There is no persistent prompt-prefix cache or automatic model download.
The current SDK path uses greedy decoding; change the instructions for a different draft rather
than expecting randomized alternatives from an identical request. Each click snapshots the latest
instructions; the original prompt remains the source until you apply a proposal and reopen the
assistant. Editing instructions now follow the source draft and explicitly override conflicting
source/image details. Requested changes to subjects, setting, length and format take precedence
above preservation. Studio clears the previous proposal while generating and flags a verbatim
repeat or unchanged source. These checks do not certify that every requested edit was followed;
the 4B model can still follow only part of a compound instruction.

The helper also accepts a local `text` command for scripts: send a JSON object on stdin containing
`requestID`, absolute `modelPath`, `systemPrompt`, `prompt`, and integer `maxTokens` (1–1024).
Optional `images` is an ordered array of objects with absolute `path` and nonempty `label`.
It returns JSONL progress followed by a result containing `text`, token counts, `truncated`,
`totalSeconds`, `imagesUsed` and available stage timings. This does not change existing headless movie/clip schemas.
The optional helper build and distribution requirements below still apply.

Qualification: request validation, token limits, RGB image patch ordering, multimodal token
alignment, process streaming/cancellation and Studio target protection have automated tests.
Local M3 Ultra checks used an existing DT 4B checkpoint, including its vision weights. An icon
description took 5.02 seconds; a two-endpoint comparison took 4.57 seconds. An eight-image test
correctly identified matching images as numbers 1, 4 and 7 in 5.60 seconds. Process peak footprint
was about 3.10, 3.14 and 4.12 GB respectively, measured by macOS `time -l` (not the RSS counter).
These are functional smoke timings with existing OS caches, not cold/warm benchmarks or physical
36 GB qualification. Text-only inference also completed, and real-process cancellation returned
in 1.07 seconds. The endpoint comparison recognized the dragon appearing but inferred some
incorrect pose/setting details; review the proposed text before applying it. Broad text quality,
small-detail/OCR accuracy and the 9B vision path remain unqualified.

### Native Qwen-Image-2.1 — experimental

Open **Generate Image…** from an asset store and set **Run with → Local MLX**. The same image
workspace supplies prompt, canvas, mood board, dimensions, steps, seed, result reuse and export.
Use **Download & prepare 8-bit model…** to select a model-library folder, or **Choose prepared
model…** to open an existing manifest. Setup pins and verifies the official
[Qwen/Qwen-Image-2.1 checkpoint](https://huggingface.co/Qwen/Qwen-Image-2.1), preserves its vision
encoder, converts the transformer/text encoder to 8-bit and calibrates the lightweight preview.
Source downloads occupy about 33 GB in addition to the prepared weights. Interrupted setup can
resume in the same folder. No Draw Things connection, cloud account or ComfyUI is required;
generation uses the app-managed Python/MLX renderer after setup.

The [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) permits
noncommercial research use. Commercial use requires a separate license from Qwen. Model terms
are separate from WeeTodd's source license.

**References:** up to ten active inputs total. An enabled canvas is image1; enabled mood-board
cards follow in their displayed order. Without a canvas, the first enabled card is image1.
Refer to inputs as `<image1>` through `<image10>` in the prompt. Disabled and zero-strength cards
remain saved but are omitted. Native cards use enable/disable, not fractional strength. Reordering
cards changes their prompt numbers. Switching to Draw Things preserves all cards and restores
that backend's settings; its existing eight-card limit still applies. Studio blocks excess active
inputs rather than silently discarding them. Prompt Assistant has a separate eight-image vision
limit and explicitly reports it when the image request contains more.

**Controls:** native flow-Euler scheduling, steps, dimensions and seed are supported. CFG is fixed
at 1. Negative prompts, generation strength, masks, custom samplers and native LoRAs are not
implemented for this engine. **Automatic** can retain request-local prefix attention; **Lower
memory** recomputes it, trading speed for memory. **Reference resolution** selects a 512 or 1024
area budget while preserving aspect ratio. Lowering it can weaken fine details. Memory preparation
is conservative and uncalibrated across hardware; it never reduces your input count or resolution
silently. The VAE is currently untiled, so large images may fail memory admission. Its allocator cache is
disabled during encode/decode and restored afterward; large convolution working sets still count
toward admission.

**Progress and previews:** loading, image/text encoding, attention preparation, sampling step/layer,
final decoding and saving report progress. Approximate latent previews update at most once every
two seconds plus the last step, without reloading the VAE or changing sampling randomness. Final
PNG output retains RGBA. Previews are temporary and removed on completion, failure or cancellation.
Weighted local jobs share a cancellable cross-process queue; each component unloads before the
next weighted stage. Cancellation checks run between shards, encoder/transformer layers and VAE
blocks, with a process termination fallback if a worker stalls.

Completed assets retain model/component hashes, ordered input hashes, resolved seed, preprocessing,
scheduler, cache mode and runtime identity. Reference-sheet and Ripple still-image editing can use
the same native provider preference. **Export Headless Job** writes v4 native image entries;
legacy remote-only exports remain compatible. Native entries currently capture explicit input
files; native outputs may feed downstream remote jobs. Dynamic upstream-to-native image bindings
are not implemented. Future Draw Things Qwen support still requires verified capability/transport
mapping; discovering the model at an endpoint does not enable this native adapter there.

Qualification includes real 8-bit 512-pixel editing and a ten-reference 1024-pixel, 40-step render,
plus tiny FP32 encoder/transformer/RGBA-VAE numerical comparisons to pinned upstream references.
These are experimental results, not full-checkpoint numerical parity or a broad quality guarantee.
The ten-reference badge test retained all numbers but changed some colors. Broad identity, fine
text and transparency quality still need further review. See [implementation status](../STATUS.md)
for measured cases and remaining qualification.

### Images, clips, and LoRAs

Use the **+ → Generate Image…** action on Global, Project, or Clip Assets. The whole-window prompt
editor provides a central canvas, a separate ordered mood board (eight active Draw Things cards
or ten total native Qwen inputs), left-side settings, preparation, result preview, and headless
image-job export. Import files, drop image assets, or choose **From Assets**.

For **Draw Things**, canvas images support fit/fill placement and **Generation strength**
from 0–100%. Mood-board thumbnails have independent enable and strength controls; zero strength
omits a reference. FLUX.2/Klein currently treats positive reference weights as enabled references,
so intermediate weights are transmitted but are not a promise of proportionally reduced influence.
Multiple mood-board inputs are supported for FLUX.2/Klein and Qwen Edit Plus/2511; other supported
image routes retain canvas image-to-image support. Models remain listed when current inputs are
incompatible, and preparation reports those conflicts. Steps, CFG, seed,
sampler, shift, and compatible LoRAs/groups are editable. **Use result as canvas** explicitly starts
another edit; generation never silently replaces the input. Control images and masks are not yet
enabled. Images are added to the captured destination store without changing the
timeline. A removed destination clip cannot silently redirect the completed image to another clip.

During Draw Things image generation, **Live preview · approximate** displays streamed latent
previews when the server supplies a supported format. Sampling step messages accompany the
preview, and the finished image replaces it after successful generation. This lightweight preview
does not load a separate VAE and may differ substantially from the final decoded image. Updates
are limited to twice per second and overwrite one temporary PNG per job, removed on completion,
failure or cancellation in Studio. Previews are never added to assets or used as conditioning.
Connections/models that omit previews retain normal progress and final-image delivery. Video
previews and full-quality intermediate VAE decoding are not part of this initial image feature.

The initial canvas-plus-two-reference route was smoke-tested locally with FLUX.2 Klein 9B KV
at 512×512, four configured steps, and 65% generation strength. Ordered references and their
weights also have transport and headless-export tests. This does not qualify every image model
or DT Cloud; control adapters and model-specific reference behavior need separate validation.
The live Studio check also covered reference reordering, strength editing, enlarged preview,
explicit result-to-canvas reuse, and export. A 512×512, four-step Klein job with one canvas and
two references produced byte-identical PNGs in Studio and WeeToddCLI with Studio closed.

Image drafts now recover across restarts, separately for Global, Project and individual Clip
stores. Studio saves linked paths, prompts, settings, references, LoRA strengths and the latest
preview path in its application-support directory. It does not copy source media or save API
keys in drafts. Missing files remain linked for replacement. Reopening a draft requires fresh
settings/eligibility preparation; Draw Things also refreshes CU estimates. An old estimate is not
treated as authorization.

**Import Config…** is available with Draw Things selected in the image workspace and in the
Draw Things clip inspector.
Open an exported JSON file or paste a configuration, then choose **Preview Import**. Named
`configuration` objects and arrays of presets are supported. The adjacent **Draw Things presets**
link opens the [official preset directory](https://github.com/drawthingsai/community-models/tree/main/configs);
each preset's `metadata.json` can be loaded here. Review the model, LoRAs, settings and omissions
before applying. If the preset names a model absent from the connection, explicitly choose an
installed model in the preview—for example, the same family in another precision. Studio never
silently substitutes model files. Prepare checks the resulting task, inputs and model together.

This initial importer supports model, prompt/negative prompt, dimensions, steps, CFG, seed,
sampler, generation strength, Shift, video FPS/frame count, Audio Shift, and whole-model LoRAs.
It accepts the `fpsId` and `shiftForAudio` aliases. Unspecified settings remain as they were;
an explicit empty LoRA list clears assignments. Prompt replacement is optional. Unsupported
settings, including controls, masks, High Res Fix and specialized adapter modes, are listed
and require explicit acknowledgment before omission. An imported preset is therefore not a
promise of complete Draw Things configuration parity.

Additional local M3 Ultra checks completed Krea 2 Turbo canvas I2I at 512×512/eight steps with
35% and 75% generation strength, and Klein 9B KV mood-board-only generation with two references
and two compatible LoRAs at different strengths. The Klein request completed in 11.5 seconds;
this is a functional smoke test, not a speed or broad image-quality benchmark. Cloud image-input
qualification remains pending; previous Cloud video validation does not establish image parity.
Live Studio testing also verified config preview/omission acknowledgment, an explicit Q6-to-Q8
model choice, LoRA-group saving, and quit/reopen recovery of both references, the prompt and
sampling/LoRA settings. Preparing the recovered request displayed 327 estimated CU for the
self-hosted route, and generation saved its result in Project Assets.

Draw Things credentials remain in macOS Keychain. Studio reuses a successfully authorized read
in memory for the current app session, including concurrent catalog/estimate/generation requests.
Editing or removing a credential clears its cached access. **Clear Session Access** in Draw Things
Connections clears all cached credentials without deleting saved keys; use it after changing a key
outside Studio. Quitting also ends the cache. Missing credentials and denied reads remain retryable.
Keychain reads, saves and removals run off the UI thread. Cancelling a job while it waits for Keychain
prevents submission after the credential request returns.

If macOS asks for Keychain access, **Always Allow** authorizes that app identity to retrieve the
specific saved item. Development builds signed ad-hoc can acquire a different identity after an
update and require authorization again. For consistent identity across builds, install an Apple
Development certificate for local development or a Developer ID Application certificate for direct
release distribution, then build with the same certificate and bundle identifier:

```bash
security find-identity -v -p codesigning
python3 scripts/build_studio_app.py --configuration release \
  --drawthings-distribution studio/.build/drawthings \
  --signing-identity "<certificate name or SHA-1 from the list>"
```

The successful choice is saved locally in ignored `studio/.build/studio-signing.json` and reused
by later builds. `WEETODD_STUDIO_SIGNING_IDENTITY` overrides that saved choice; an explicit
`--signing-identity` takes precedence over both. An unavailable selected certificate fails the build
and preserves the previous bundle; it never silently falls back to ad-hoc signing. Passing `-`
explicitly selects ad-hoc signing again. A first build without a configured identity still supports
ad-hoc signing and prints a warning. The build signs nested executables before the bundle and
verifies the result. This configures signing, not notarization or App Store distribution; it does
not broaden Keychain access permissions. Switching from ad-hoc to certificate signing may require
one new authorization.

Use the timeline **+ → Draw Things** to create a video clip. Refresh models, select an exact server
model, write its prompt, and prepare it. Native MLX recipe files are not needed for this provider.
Setup follows **Generation → Draw Things → Task → Connection → Model → LoRAs / Groups**. Tasks narrow verified
connections and models using their advertised input combinations. Unverified connections remain
available with **refresh to verify** until Studio has their catalog. Changing the task clears a
verified incompatible model selection; changing the connection clears the model selection.
Frame images and saved settings are retained. LoRAs and groups are filtered by the selected model.
Dimensions use a 64-pixel grid. Generation FPS must be an integer; LTX frame counts round upward to
`8n+1` to cover the requested duration. Movie finishing applies the project/clip output settings.
Generation adds an audiovisual movie to version history and Clip Assets. A changed clip is not
marked current by an older render finishing later.

**H3 first and last frames:** select **First and last frames** and drop images onto the timeline's
**FF** and **LF** slots, even before selecting a connection or model. Then choose a discovered
**MiniMax H3 FL2VA** model. You can also use **Use in clip → First frame** and **Use in clip → Last frame**
in Media & Assets. Selecting an incompatible model preserves your images and reports the mismatch;
Prepare Clip requires a compatible model before generation.
Studio sends the first image as Draw Things' canvas input and the last image as its first enabled
mood-board (`shuffle`) hint. Both images are center-cropped to the generation dimensions and hashed
before submission. The last endpoint follows the resolved frame count when duration changes.
There is no need to arrange the canvas or mood board in the Draw Things app.
After generation, endpoint clips adopt the resolved duration (124 / 24 = 5.167 seconds for a
five-second H3 request) so the timeline and headless movie finishing preserve the final frame.

H3 uses 24 FPS and `17n+5` frames (five seconds rounds up to 124 frames). Its defaults are 50 steps,
DDIM Trailing, CFG 1, Shift 12, and Audio Shift 3; steps and both shifts remain editable. H3 video
and 32 kHz stereo audio stay together. Only models actually advertised by the selected endpoint are
offered; a model installed in the local app is not necessarily available through the cloud API.
LTX still supports first-frame input only through this adapter. Last-only and arbitrary middle
keyframes are not enabled. H3 Ref2VA supports 1–9 still-image references through its separate
**Image references · H3** task. A conflicting attachment/task is
reported before generation rather than discarded. Headless exports use the same image contracts.
Clip Assets are storage; only items listed under **Conditioning** are generation inputs.
Keep previous rendered movies in Clip Assets without attaching them as a Reference to a Draw Things clip.
Prepare Clip names unsupported attachments, and existing unsupported inputs show an inline warning.
Remove the attachment with **×** to retain its media in the asset store. Draw Things input menus
offer the selected task's supported endpoints or H3 image references; attachment strength is fixed
at 1 (LoRA strength remains editable).

**H3 Turbo:** import a compatible Turbo LoRA into Draw Things, then click **Refresh** in Studio.
Enable it under **Server LoRAs**, set its strength, and edit **Steps**. A local FL2VA test used
`minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16` at **0.6 strength**, **4 steps**, DDIM Trailing,
CFG 1, Shift 12, and Audio Shift 3. It completed at 768×448 with 124 frames and 32 kHz stereo audio
in 226.8 seconds on an M3 Ultra. First/last images closely matched the supplied endpoints with
reconstruction differences. This is one local test, not a quality or speed guarantee for other
LoRAs, hardware, resolutions, or prompts. Studio references the server's installed LoRA and does
not copy its weights.

Start remote LTX clips at **24 or 25 FPS**. The pinned Draw Things decoder produces audio on a
fixed causal clock, independently of playback FPS. Its complete audio can end slightly before the
last picture: a 121-frame clip contains 4.81 seconds of audio, versus 5.042 seconds of video at
24 FPS. The helper and Python bridge independently verify the exact causal sample count, then
preserve the original frame rate and soundtrack without stretching either. The remaining picture
plays after the soundtrack ends. Other audio still uses the one-frame duration check; missing or
incomplete audio is rejected. This exception does not permit audio extending past the video.

Earlier helpers could report a receiving/finalization error after all frames and audio arrived.
Update both the helper and renderer (refresh an app-managed runtime if applicable). Preserve the
failed job's `render/media` folder before retrying: complete received files may be recoverable
locally without another cloud generation. A missing completion manifest is not proof that the
remote request failed or that another submission would be free.

The pinned helper recognizes selected FLUX, Qwen Image, and Z-Image model families for images, and
LTX 2/2.3 and H3 FL2VA/Ref2VA for video. Only exact endpoint IDs advertised by both the helper and server
appear. Remote LTX 2.5 remains unsupported; use its native engine.

One **First Frame** image is supported for LTX video. Its full file hash participates in preparation
and request identity, and the helper rechecks it before submission. Orientation is respected and the
image is center-cropped to generation dimensions. H3 FL2VA also supports a First Frame / Last Frame
pair, as described above; H3 Ref2VA uses the separate still-reference task. LTX last frames and
references, interior keyframes, movie/audio references, audio drivers, control hints and native
clip extensions are rejected explicitly in this remote adapter.

Server LoRAs are filtered by exact remote model compatibility. Strength ranges from 0 to 2. Named
remote groups copy their members/strengths to a clip and remain separate from the native LoRA library.
Before discovery, or after a failed refresh, saved models and LoRAs are marked unverified rather than
unavailable. Saved LoRA strengths remain editable. For local servers, enable the gRPC API and Model
Browsing in Draw Things, then Refresh in Studio. Only a successful catalog can mark an assignment
unavailable for that connection/model; generation always revalidates it.
Only ordinary LoRAs with matching SDK model family are advertised; specialized modifiers, alternate
decoders, and unverified variants are excluded. Importing local SafeTensors as a remote LoRA is not
supported: this adapter has no verified converter/upload workflow.

### Headless jobs and qualification

Movie/clip exports containing Draw Things work use `weetodd-studio-job-v3`. Image jobs can be exported
from the image prompt editor. Jobs remain sequential and refresh eligibility before each request.
Image-to-video dependencies bind a completed image as the first frame before the video is estimated.
See [headless examples and credential setup](../examples/headless/README.md).

Resume reuses only verified artifact hashes. A recorded submitted/completed remote request whose
artifact is unavailable is never automatically regenerated; create a deliberate new job/output after
checking the earlier attempt. Closing Studio is supported. Self-hosted jobs still require their server;
Cloud API jobs do not require the Draw Things app.

| Path | Qualification |
| --- | --- |
| gRPC discovery/image transfer | Synthetic server and Studio image UI tested |
| Video + separate audio | Synthetic gRPC, Studio clip UI, and real FFmpeg timing/publication tested |
| First/last frame / LoRA contracts | Automated wire mapping, compatibility, hash, duration/resume, status, and failure tests; real local H3 FL2VA four-step Turbo render verified from both the shared renderer and packaged Studio |
| CLI image-to-video and movie assembly | Synthetic server with Studio closed; real FFmpeg clip, dissolve, title, and supplementary-audio assembly tested |
| Completed-job resume | With fixture server stopped, reused both remote artifacts and the same final movie hash |
| ComfyUI image/video/estimate workflows | Saved, fixture-bound API graphs executed in isolated ComfyUI; repeated estimates refreshed and image output saved |
| Packaged Studio | Bundled helper discovery, connection test, prompt CU, project reload, and light/dark appearance checked; H3 endpoint render marked Generated after restart and exported with all 124 frames |
| Helper corresponding source | Distributed archive extracted and rebuilt against its supplied editable dependencies |
| Native project/job compatibility | Focused regression tests |
| Real Draw Things model generation | Local H3 FL2VA: 124 frames at 768×448/24 FPS plus 32 kHz stereo audio; LTX 2.3 Cloud: 81 frames at 768×448/25 FPS plus 48 kHz stereo audio |
| Direct Cloud free-tier generation | Saved-key verification, free-allowance check, Prepare and Generate completed in Studio with PAYG disabled; unknown allowance still fails closed |
| DT+ App Bridge generation | Unavailable pending verifiable billing policy |

Fixture tests establish software behavior, not output quality or a promise that a particular remote
model will fit a free-tier allowance. Retail signing/notarization and clean-Mac qualification remain
separate release work.


## Project subjects and shot list

Open **Movie → Shot List…** or the **Shot List** toolbar button. The project stores this optional
planning document alongside existing media and timeline data; older projects still open unchanged.

1. In **Original script**, paste the complete brief/script, including dialogue, camera and sound.
   **Identify characters, props and locations…** opens the local Qwen subject workflow. Complete the
   step and choose **Add to project**. Subjects that you approved retain description approval;
   the remaining subjects begin as drafts. Review completeness: extraction can miss subjects.
2. In **Characters, props & locations**, edit each name, kind, aliases and appearance. Source evidence
   and suggestions are read-only. **Approve description** locks that version. Unlock it before editing. Add
   subjects manually or merge duplicate subjects of the same kind; both must be unlocked. Merging
   keeps the selected destination's description and preserves the other description as a review note.
3. **Import images…** or **Choose project/global image** links existing character sheets, prop views
   or setting images into Project Assets without copying media. **Approve references** is separate
   from approving a description. Missing, relinked or modified files require reference review.
   Images already made in Studio's Draw Things editor can be selected here. **Create character sheet…**
   opens structured Character Director for character records; selecting a candidate and approving its
   reference remain explicit review steps.
4. In **Workflows**, choose **Add to project** after movie clip planning to import the detailed shots.
   Repeating an import adds missing records and preserves existing edits/approvals; it does not
   refresh an existing shot from a changed workflow result. The original source text is preserved.
5. In **Shots**, add/reorder/edit shot names, frame counts, action, detailed direction, first/last
   descriptions, camera, dialogue, sound and subject links. FPS is the shot list's own planning
   timebase. Approving a shot checks required descriptions, timing, subjects and continuous-shot
   boundaries. Subject or timing changes make dependent approvals stale. Changes are undoable and
   autosaved with the project. Timeline clips and movie settings are not changed.
6. **Export shot list…** writes a planning JSON document for inspection or future automation; it is
   not an executable generation job. Subject and shot UUIDs remain stable. Explicit subject
   description approvals carry into newly imported project subjects. Reimports preserve existing
   project edits and approvals; they do not overwrite them with workflow changes.

Subject workflow results open as a hierarchy of **Characters**, **Environments**, and **Props**,
with names sorted within each group. The new **Identify and review visual subjects** workflow
adds an internal description review after extraction. Each subject is checked against five visual
aspects appropriate to its type (for example, environment layout, materials, lighting and features).
A drafting call proposes a fuller design, and a separate critique checks only the current candidate.
There are at most two rounds per subject (four model calls); unresolved or malformed reviews remain
**Needs attention**. The agent never grants human approval.

### Character Director

Open **Director → New Character Director** for a dedicated window without a movie project.
**Create character sheet…** opens the same editor wherever Director or project subject review offers
character references. Documents autosave separately from movies; **Recent** reopens saved characters,
and **Export… / Import…** moves a folder containing the document and its media. A corrupt document
reports an error instead of being replaced. Exported media do not grant subject or reference approval.

Edit individual fields with common-value menus, custom values, Required flags, and field states: Value,
Unspecified, Absent and applicability-gated N/A. Empty fields are omitted. Required species, character type and rendering
preset start enabled. Clothing, accessories, distinctive features and surface descriptions have
repeatable records. Undo/redo preserves the document's revision history for stale-analysis checks.
The prompt is read-only and compiles in this order: layout, identity, body, face/head, hair, clothing,
accessories/features, materials, style, camera, lighting, consistency and composition. It begins exactly:

```text
4-view turnaround of a character, front view, side view, back view, facial close-up, plain solid white background
```

**Photograph** is a versioned photography preset with anatomy-appropriate skin, hair and material
clauses, rather than a single appended word. Cinematic Photograph, Realistic 3D, Stylized 3D, Anime,
Comic, Oil Painting, Concept Art, Clay and Sculpture use the same appearance fields. Character image,
style image and authored-text analysis use the configured app-owned Qwen3.5 4B model; character and
style may share one image. Bounded calls propose forensic visible details with evidence and uncertainty.
Descriptive values include supported shape, location, proportions, texture and variation in one or
two clauses; simple facts can remain short. Evidence and uncertainty stay separate and concise.
Smaller field batches reserve room for detail without raising the 1,024-token output limit.
Schema repair preserves independently validated descriptions when repairing a different field.
Malformed inventory JSON gets one correction pass using the same source and deadline before any
attribute extraction. A second invalid response stops safely without applying partial records.
Physical height/scale measurements come from manual fields or authored text, not guesses from pixels.

Apple Vision selects the foreground instance containing one unambiguous face, removes the background
onto white, crops around the subject and prepares an orientation-correct overview capped at 512
pixels per edge. Character Director shows that prepared image and its isolation status. The original
file is retained. Head/scalp and torso/lap details use the same isolated subject, each capped at 512
pixels; only the corresponding face/hair or wardrobe calls receive them. Missing or ambiguous
subject masks fall back to a bounded original overview with a visible diagnostic, never a claim of
successful isolation. Character detection and mask/crop preparation use a preview capped at 1600
pixels, with streaming hashes and source-change checks before publishing proposals. Style analysis
retains the original scene context and uses the helper's bounded whole-image input.
A bounded visual inventory assigns distinct items to record slots before field extraction, including
single optional feature slots, omits unused slots and binds material records to named targets. Clothing
analysis carries garment identity into material calls,
omits unobserved footwear and requires literal evidence for condition: fading and creases must not
be generalized into damage. Structured or negated evidence is checked before a proposal can be applied;
model observations still require human review.
Scalp coverage, hairline, residual hair length and style are analyzed together. Dedicated visible
skin-tone and skin-texture fields preserve observations without inferring ancestry; ancestry remains
authored information. The compiled hair section places coverage before residual hair details.
Apple Vision separately uses bounded previews for panel detection
(1600-pixel edge) and reference-head masking (2048-pixel edge), mapping crops back to source pixels.
Review and select proposals before applying them. Accept selected reports the applied count and retains
unselected or rejected values for further review, including after reopening the document. Character
analysis remains applicable after unrelated style-reference, rendering, or Required-flag changes.
Edits to its appearance fields or source image still require a fresh analysis; style and authored-text
analysis likewise protect their own inputs. Older saved batches without an input snapshot keep the
conservative document-revision check and must be rerun if marked stale. Importing a reference alone
does not clear accepted fields. Invalid and unsupported proposals remain unappliable.
Image analysis does not infer authored demographic identity fields. Legacy descriptions
remain available for mapping; legacy character-sheet prompts must be mapped before regeneration.

The initial recipe requires **Draw Things Local**, installed **Krea 2 Turbo**, and its compatible
four-view LoRA, starting at eight steps, CFG 1 and **1920 × 1088**. Discovery selects an unambiguous
installed pair; missing or ambiguous models/adapters require explicit selection.
Enable **Model Browsing** in the Draw Things local server settings so Studio can discover its models
and LoRAs, and turn **Bridge Mode off** for local inference. A localhost connection can still forward
jobs to Cloud Compute when Bridge Mode is enabled. The sheet requests four panels in one row.
Apple Vision preserves separate foreground instances, combines body detections and measures image
gutters on a bounded preview, then maps crops back into source pixels.
It never silently substitutes equal quarters. Review actual
crop boundaries and roles; uncertain detection supports manual crop creation/correction.

For detail refinement, each approved panel is cropped, enlarged exactly **2× with Lanczos**, and white-padded to multiples
of 64 for **FLUX.2 klein 9B** with **HighResolution9B** (also recognizing the spelling
**HichResolution9B**) and `high quality` in its managed prompt.
Discovery prefers an unambiguous installed **9B KV** variant for new documents; explicit saved
model selections are preserved. Panels render serially. The combined outputs are reassembled at **3840 × 2176**, removing transport
padding while preserving the reviewed crop placement. Four equal-width crops would use 960 × 2176
inputs; detected crops may have different widths.

A separate **Reference face** input uses local Vision background removal, retaining the original,
head crop, mask, RGBA cutout and white-matted transport image. Ambiguous faces require a manual crop.
**Replace faces** enables the exact `bfs_head_v1_flux-klein_9b_step3750_rank64` adapter and begins each
head-swap prompt with `head_swap: replace the head with the reference head.` The target panel is reference
1 and the head is reference 2. BFS may replace hair as well as the face. A combined head/detail
pass remains available for comparison. New documents default to **BFS, then HighResolution9B**:
only BFS runs at the **original panel resolution**, padded to multiples of 64. Its output is cropped
back to the original panel bounds, enlarged exactly **2× with Lanczos**, then padded for the
**HighResolution9B-only** detail pass. BFS never receives the enlarged input in this separate recipe.
Each FLUX pass defaults to **eight steps**.
Existing saved recipe choices and step counts are preserved; select the separate recipe and eight
steps per pass to compare an older document. LoRA strength is shown as a percentage: **80% = 0.80**,
independent of image strength and CFG. New documents use versioned minimal prompt presets:

- **BFS — Trigger only:** `head_swap: replace the head with the reference head.`
- **BFS — Trigger + quality descriptors:** the same trigger followed by
  `4k, realistic skin texture, realistic hair.`
- **Detail — Trigger + quality descriptors:**
  `High Resolution. high quality, realistic skin texture, realistic hair, 4k.`

BFS defaults to trigger-only. The minimal presets omit the character fields, Photograph preset,
wardrobe descriptions and camera instructions; the ordered images supply the edit context.
The HighResolution prompt style is shared across 4B/9B, while this recipe still requires the installed
9B adapter. Initial-sheet generation retains the full structured prompt. Existing documents keep
**Legacy descriptive prompt** until a preset is explicitly selected; saved strengths, prompts and
historical outputs are preserved. Detail-strength changes reuse matching BFS stages. A changed BFS
prompt invalidates its own stage and dependent detail. Presets do not guarantee likeness or realism.

Progress and available live previews appear in the window. Cancel stops the active local operation;
completed panels are retained and reused when their execution inputs and output hashes still match.
Interrupted submissions are marked uncertain and require inspecting saved output and explicitly
allowing retry, preventing automatic duplicate submissions. Closing a busy window offers Keep Running
or Cancel Job. Render settings, ordered input hashes, head preprocessing and individual panel takes
remain in the final sheet's provenance. Choose **Use reviewed candidate**, then approve references separately.

Earlier combined-pass qualification exercised initial Krea generation, all four serial Klein/BFS/detail passes,
live previews, head masking, assembly and completed-panel reuse. One 8-bit, four-step-per-panel run took about 101 seconds
for the initial sheet and 778 seconds for refinement; a sampled Draw Things process footprint during
the largest panel was about 23 GB. These are observations from one Apple M3 Ultra with 256 GB, not
general speed or peak-memory guarantees. Head consistency and composition still require visual
review. A follow-up eight-step KV run on the same machine completed all eight separate head/detail
passes in 600.5 seconds, with 40 live preview revisions. BFS used native panel dimensions before
2× detail preparation. The front view retained a balding scalp, but the close-up still invented top
hair and exaggerated skin texture; this qualifies execution, not general likeness or realism.
A September 21 minimal-prompt comparison used eight-step KV passes, BFS at 1.0 and
HighResolution9B at **0.80**. The corrected initial sheet and refinement retained a plain crew-neck
shirt and blue jeans without the previous invented tears or shirt placket. The front pair took
105.1 seconds; the remaining six passes and assembly took 353.6 seconds, reusing that pair exactly.
This run failed visual acceptance: BFS placed a forward-facing head on the back-view body, and the
close-up retained exaggerated skin contrast and changed lens tint. The orientation error was already
present before the detail pass. Review native BFS outputs as well as the final assembly; shorter
prompts and 80% detail strength do not establish reliable turnaround geometry or photorealism.
The 0.80 comparison is a test setting, not a change to the generic 1.0 detail-strength default.
The installed model/LoRA combination must pass Draw Things preflight before rendering.

**Create reference…** remains available for other subject types in workflow and project review.
It opens the existing Draw Things image workspace with editable character turnaround, portrait,
prop, environment, set, wardrobe and custom-reference templates. The prompt uses the current
saved description and linked object definitions. Apply a template to rebuild the prompt, then
edit it freely. Pose/camera instructions are prompt directions, not a guaranteed pose-control
adapter. The previous ordinary image workspace is restored when you return to approval.

Choose any image model recognized in the connection's model catalog, including custom checkpoints
using supported DT image families. **Starting settings** provides the user's Krea Turbo eight-step
starting point and a Klein four-step/CFG-one starting point; it never switches models or inserts
LoRAs. Qwen Edit and other models can use editable steps, CFG, sampler, shift, compatible LoRAs and
groups, or **Import Config…**. Models introduced after the bundled DT client may require a client
update. Unsupported input combinations are reported instead of silently dropping references.

The image editor loads the selected connection's model catalog automatically when opened or
when the connection changes. Loading, empty catalogs and failed discovery have separate messages;
**Retry loading models** / **Refresh** retries discovery. A failed refresh retains the last loaded
list with a warning. Clearing the model selection does not clear that list. Restoring a draft or
importing a config preserves its model and LoRAs; explicit connection/model edits apply compatibility
changes. Models remain listed even when the current canvas/mood-board inputs are incompatible.

The reference editor opens as a larger, resizable window. **Reference setup** expands the template,
style and pose/camera controls; **Existing references** expands reusable image inputs. The **Mood
board** button shows or hides that panel (hidden initially when empty). **Tools** contains prompt
assistance, config import and headless export. Config and connection dialogs open on the current
editor. The zoomed canvas scrolls within its viewport, without covering surrounding controls;
**Fit** resets the view and **Inspect image** opens the original-image preview. **Hide prompt**
gives the viewport additional vertical space without changing the saved prompt.

Existing subject references appear as thumbnails. Choose **Canvas** for an image-to-image input,
or **Mood board** for supported reference editing. Klein/FLUX.2 and Qwen Edit Plus/2511 expose
multiple mood-board inputs; Krea and other basic routes retain canvas conditioning. References
are never silently enabled, and canvas strength remains adjustable. Check Settings & CU before
generating. No cloud job is submitted by opening the editor.

Click a reference thumbnail to inspect its original image in a large popup. **Fit** shows the
whole image, **Actual Size** uses one image pixel per display pixel, and the zoom slider allows
closer inspection with scrolling. **Done** or Escape returns to the same review. This is available
in workflow approval, project subjects, the reference generator and the mood board.
References are labeled by subject and reference number, and the Candidates menu numbers each
result. Preview selection uses the full file path independently of those labels, so files named
`00000000.png` in different generation folders remain distinct. Removing or reordering a
reference does not retarget an already selected image. Original files are not renamed.

Generated candidates are new Project Assets, retaining the source description, template and
generation provenance. **Candidates** recalls earlier results for the same object. **Use as
reference** links the selected result without copying its file or running Qwen; approval remains
your decision. Workflow attachment invalidates affected approvals and marks older agent notes as
outdated. Project reference approval stays separate from description approval. The per-object
draft is saved for reuse, and an attachment failure leaves the generated asset available.

Select a subject and use **Add reference images…** to attach up to eight existing images.
**Review & improve description** uses the installed local Qwen3.5 **4B** to inspect those images
alongside relevant script passages. The 9B text route does not support these visual reviews.
Image observations appear under **Observed in reference images**; invented additions appear under
**Proposed design details — approve or revise**. These are model interpretations for you to verify.
Source-labeled phrases must occur in their cited text; a valid citation number alone is insufficient.
Unmatched paraphrases or added details are conservatively labeled as proposals for your approval.
When no images are attached, the writer is explicitly told to use source or proposal details.
If design proposals are allowed, it writes five plain visual aspects and the app labels the
result as proposed design details. The separate critic still checks identity and visual coverage.
The app also flags known inventory names used inside another object's appearance, prompting an
object-ID reference instead. This prevents a missed named-object conflict from receiving a clean
agent report; it does not detect every paraphrase or settle ambiguous ownership automatically.
Any hallucinated image claim is retained only as an unverified design proposal, never as an image
observation. Zero is not a valid source/image citation and is never reassigned to a real attachment.
A reference supplements the script; conflicts should be flagged rather than silently settled.
Images stay referenced in place, and **Add to project** links them into Project Assets for reuse.
Selecting images alone does not run the model: press the review button to apply them.

Edit the name and description, then choose **Save & approve description**, or use **Save changes**
followed by **Approve description**. Agent notes are advisory: you can approve your own corrections
even when the agent flagged issues or reviewed an earlier version. Another model call is optional,
and approval does not run it. The saved description is locked after approval; unlock it before editing.
The agent report stays intact and is labeled when it predates your edits or image selection.
Project-level approval follows the same human decision model; imported notes and warnings remain
visible. Director also offers one explicit batch approval for the saved subject inventory; save
unfinished edits first. Individual approval remains available. Unlock a description before changing
it. IDs are app-controlled, and evidence and
suggestions are selectable, read-only text. **All outputs (JSON)** is an optional inspection view.
The project subject list uses the same grouping and read-only source fields. Existing checkpoints
can be reviewed without regenerating their subject inventory.

Subject extraction reads numbered source passages in bounded sections, selecting evidence IDs.
Code copies the original passages instead of accepting model-authored quotations. Legacy extraction runs at most
16 sections across three subject kinds (48 calls per attempt), with one retry, and at most 24 final
subject proposals. A section that reaches its eight-proposal cap adds a visible completeness warning
in both the workflow and imported project. Guided movie and music-video extraction supports 64 final
subjects, requests eight new names per page, and makes at most nine page requests per kind and source
section. Bounded oversized responses retain all validated names; stalled pagination fails explicitly.
A uniquely matching whole name can repair a wrong paragraph citation with a review warning; ambiguous
names require corrected citations. Subject IDs are host-assigned, so accented names do not
break the machine-readable contract. Each request has the existing 1,024-token response ceiling. Files/models stay
referenced in place; no image generation, cloud CU spending or model download occurs in this stage.
Evidence selection and descriptions still need human review; exact source text does not certify
that a model interpretation is correct.

Use **Create reference…** to generate and inspect candidates, assign first/last images to shots,
then use **Add approved selection to timeline**. Clip preparation validates the selected engine's
task and frame-grid requirements. Automated versioned sheet/endpoint generation remains future
work. Current workflow checkpoints remain available independently.


## Production library and object relationships

Open **Movie → Production Library…**, the matching **Media & Assets** button, or the library
button in **Shot List**. This catalog stores small metadata packages in SQLite using the macOS SDK;
there is no new Python dependency or media/model copy. Existing projects remain portable JSON and
open without the global database.

The **Production objects** tab groups Characters, Environments, Sets, Locations (unclassified),
Props, Clothing and Outfits. Existing locations keep their IDs and meanings until you explicitly
change their kind. An environment defines shared architecture/design; a set must select its parent
environment. Day/night, weather and temporary states belong in the shot’s appearance notes rather
than duplicated environments.

- Edit names, descriptions and comma-separated tags. IDs remain app controlled.
- Use **Link object…** to reference another definition. Choose Contains, Wears, Holds, Uses,
  Located in or Part of, and add a placement/state note. Multiple placements can reference the same
  prop ID; their instance IDs remain separate. Clicking the linked name opens its definition.
- **Publish from this movie** saves an immutable version with linked dependencies and image metadata.
  Publishing an environment includes its sets; publishing one set includes its parent and required
  objects but excludes sibling sets. Unchanged publications reuse the current version.
- Search global packages by name, kind or tag, then choose **Use in movie**. The movie stores pinned
  definitions and image links. Existing conflicting IDs cause a clear error and leave the movie
  unchanged; automatic version replacement/three-way merging is not implemented.
- Edit an imported object locally to make a movie variation. Its library origin remains visible;
  publishing creates another package version and never silently updates other movies.
- Under a shot’s **Resolved objects and references**, add shot-only appearance/state overrides.
  They do not rewrite the movie or global definition. **Add approved selection to timeline** includes
  these notes with the resolved object descriptions in each new clip's prompt; later edits do not
  rewrite already applied clips automatically.

Approvals include transitive object-definition revisions. Changing a linked coat or prop makes
its owners’ description/reference approvals and affected shot approvals stale. Cycles are traversed
safely; missing IDs, invalid environment parents and excessive relationships fail structural checks.
Human approval remains independent of optional agent reports. Referenced objects cannot be deleted
until their uses are removed. Merges remap links transactionally or report a conflict.

**Export package…** writes `weetodd-production-library-v1` metadata for sharing/import into another
movie. **Export shot list…** writes `weetodd-shot-list-v2`, including planning records, reference asset
metadata, resolved dependency snapshots, shot appearance overrides and unresolved-link warnings.
Neither export embeds image bytes. Use the movie’s **Collect Media** feature for portable files;
missing paths must be relinked before generation. A catalog package is limited to 2,000 objects and
2 MB of metadata; searches show at most 100 matching packages. SQLite revisions are whole dependency
snapshots per published root, retaining stable object/placement IDs across versions.

The reviewed inventory workflow v1.2.0 now runs extraction → **Link reusable objects** → description
review → **Review object coverage**. `project.link_subjects@1` proposes links only to known inventory IDs, keeps source evidence,
and separates suggested missing objects from the inventory. It accepts up to 64 subjects and two
attempts each, with per-subject checkpoints. Review or edit relationship roles/placements before
approval. Old saved workflow definitions stay pinned; choose the current builtin for a new run to
include the new linking step. Runtime-ready local model bindings remain outside portable definitions.

Automatic sheet workflows and endpoint-frame population remain future work. Use **Create reference…**
for explicit reference-sheet generation, assign the selected frames to reviewed shots, and use
**Add approved selection to timeline** to create their clips.
Explicit library-version reconciliation, saved appearance
presets and automatic classification of legacy locations are follow-on work.

Qualification on 2026-09-13: 116 Swift tests (one optional skip), 184 targeted Python tests, native
package import/publication/search/navigation checks and a local Qwen relationship smoke test.
The latter completed in 12.01 seconds with three calls: a valid actor→jacket link was retained,
while a reversed jacket→actor wearing relation was rejected and left for review. Advisory “ready”
means the proposal passed available checks, not that its meaning has been independently proven.


Descriptions in workflow review and the project object editor highlight explicitly linked object
names, aliases and IDs as native macOS links. Hover to read the target’s current description;
click in a locked description, or Command-click while editing, to open the object. The text remains
plain text in project/workflow files. Ambiguous shared names are left unlinked; the target’s explicit
ID can be used instead. Relationship-row names also show description tooltips.

The final `project.review_object_coverage@1` pass checks the enriched descriptions against the whole
inventory and a bounded movie/global library shortlist. It adds validated draft relationships and
exact phrase-to-ID anchors, so wording such as “her jacket” can link to the named clothing object.
Changing the description invalidates its semantic anchors until it is reviewed again. Invalid
proposal entries are flagged without discarding independent valid links. Ambiguity and missing
distinctive objects remain visible for human review; the model cannot invent target IDs or approve
descriptions. Approved rows and their linked dependencies remain unchanged, with proposed changes
shown separately. Unlock the relevant descriptions and rerun coverage to apply those proposals.

Use **Find missing object links & reusable matches** on existing completed subject checkpoints to
run this pass without repeating extraction. Saved execution inputs remain frozen; refreshing the
review catalog does not invalidate the original job. The pass processes at most 64 objects, with
two attempts each and saved per-object progress for resume. The catalog contains at most 64 metadata
records; each model request uses at most eight compatible library candidates.

Library matches show their definition, scope, version and reason. Choose **Use this definition on
import** to reuse its stable ID when adding the results to the movie. Studio rechecks the source
proposal and the selected definition/dependencies before import; stale selections need review again.
**Create draft in movie** adds a suggested missing object for editing and approval. Neither action
copies images or model weights, and missing-object creation does not silently change the workflow
inventory. Reference generation and applying approved shots to the timeline are explicit actions;
automatic sheet generation and reference verification remain future work.

## Guided movie planning

**Movie → Workflows… → Create a movie** begins with a creative brief. Set the finished movie length,
look, widescreen/vertical/square framing, camera feel, sound and permission to propose missing
details. Every preference also accepts your own wording. Frame rate starts from the current movie;
technical timing controls are under **Advanced timing**. Reference images remain linked files and
are observed locally. This workflow does not change the movie's render settings by itself.

The agent asks at most six consequential follow-up questions. Save partial answers when needed;
approval stays unavailable until every required answer is present. Identity questions need an
explicit answer. Other creative questions can be delegated to the director. The original story,
question wording and evidence remain read-only. Selected answers are passed downstream without
the rejected options, and style/camera preferences are kept out of extraction's story evidence.

New guided plans (definition version 1.1.0) have three required review stages:

1. **Brief:** review the source, creative preferences and consequential questions. Identity ambiguities
   still require your explicit answer. References are observed once and retained as evidence.
2. **Subjects:** review classification, descriptions, relationships, references and reusable library
   choices together. Correct object types without changing IDs. Discovery copies checked source
   phrases; proposed visual details remain separate from evidence. Classification receives each
   object's established description, rather than guessing from its name alone. Unresolved agent
   notes remain visible. Save edits, then explicitly approve the saved inventory as a batch or
   approve individual subjects. **Use this definition** pins a reusable appearance; final import
   rechecks the actual library version and dependencies.
3. **Shots:** review the story and timed actions with visible start/end states. Approve the shot plan,
   then finish the deterministic endpoint checks and prompt compilation before **Add to project**.
   Approved character descriptions are retained in full rather than silently shortened.

Detailed review adds separate stops for classification, inventory, visual design, treatment and
prompt preview. Intermediate steps and technical records remain inspectable in either mode; no
model report grants human approval. Old saved jobs keep their original stops instead of being
silently migrated.

Shot correction offers an explicit scope: action, states, location/connection, characters or the
whole shot. The existing shot is supplied as context, and host code retains fields outside the
selected scope. IDs and timing remain app-controlled. Invalid continuity stops for a correction
instead of silently changing an unrelated field. Unaffected shots retain their saved choices.

The optional H3 preview shows all three prompt fields, resolved subject IDs and reference asset IDs. It is
not a render recipe: actual model/task support, dimensions, frame count and image roles still need
validation when a generation job is created. Image references do not imply first/last-frame binding.
Speech requires explicit wording and speaker/language assignment; the preview does not silently
invent or assign dialogue. Music is omitted unless requested. No images, movies or cloud jobs are
generated by these stages.

App/helper rebuilds, runtime changes and output-token changes do not automatically regenerate
completed approval-workflow steps. Their saved descriptions and approvals are reused when the
step definition, story inputs, upstream results and reference-image contents are unchanged.
The original execution provenance stays with each completed result. To replace completed work,
explicitly unlock and regenerate it; normal resume continues from the saved review.

Every step must complete and every required approval must remain valid before guided import. Editing prior decisions invalidates
dependent results. Saved jobs retain their definitions: **Run remaining** on an old extraction-only
job cannot add missing stages. Its **Start guided workflow…** button starts a separate job using
the original story. It does not overwrite the old review.


### Create a music video

Choose **Movie → Create music video** and select an imported song or a generated YuE2 take.
Enter lyrics when available, or mark the track as instrumental or lyrics unknown. Director asks
for essential missing creative decisions in the brief review. Reference images and existing
production-library characters, props, environments and sets use the ordinary reviewed movie
workflow. The imported project retains those objects, references, lyrics and source-song timing.

Clip timing is automatic: the default minimum comes from the native model contract and the
maximum is 15 seconds, capped by model support. Advanced controls override valid bounds. The
Audio analysis panel offers learned beats/downbeats and English word evidence, alongside a quick
DSP preview. Set up the compact analysis models or choose an existing verified model folder.
For difficult singing, opt into **Isolate vocals first**, then include its additional 35.6 MB model
in setup. Estimated vocals feed only English word evidence; beats and the movie soundtrack keep
the original song. Isolation is cached and can be cancelled.
Analyze the song, inspect words/lines and uncertain omissions/repeats, then add, move, remove or
lock suggested cut markers. **Reviewed cut…** creates an editable marker at source seconds you
enter for a lyric boundary; it preserves the model evidence. Supply full-song lyrics; unsupported
words remain untimed.
**Compare recognition with supplied lyrics** preserves the raw recognition and shows separate
lyric-assisted wording. The matcher checks short phrase boundaries and uses matching surrounding
words to propose corrections; every proposed letter still needs acoustic support. Word rows show
agreement, lyric assistance or unresolved differences. Unsupported supplied lyrics are not inserted
into the transcription. Lyric-only edits reuse cached acoustic evidence. These labels describe
model evidence, not human verification, and sung timestamps still require review.
A frame planner combines audio cues, pacing and locks while penalizing cuts inside supported words.
Section/repetition and possible vocal-break labels are reviewable suggestions, not verified chorus
or instrumental classifications. Unknown lyrics produce an unreviewed English transcript.
LTX 2.5 is the default; LTX 2.3 and H3 also support native audio-driven planning, while soundtrack
mode supports other configured video models. Draw Things planning requires the selected model's
explicit duration bounds. All generation still validates actual model/task support.

After reviewing Brief, Subjects and Shots, add the plan to the project. In **Shot List**, select
adjacent unapplied shots and choose **Combine** for one continuous take with ordered text and the
combined duration. **Restore original shots** restores their original definitions and references.
**Split…** divides one unapplied shot at an editorial frame and preserves the exact song interval,
including the natural audio tail. Review both resulting actions and their new boundary; original
direction text and outer frame references are retained. **Generation settings for selection…**
sets the model and render dimensions for selected unapplied shots, persists them with the plan,
and carries them into timeline clips. These edits mark affected shots for review.
To move a cut between music shots, adjust their frame counts while preserving the total song
length, then choose **Realign song intervals to shot lengths**. Existing timeline clips must
already match the revised starts and lengths; the action changes timing metadata, not footage
or the audio track. Review the affected shots and explicitly reuse any trimmed takes afterward.
Realignment requires one continuous song and uncombined shots, preserves the natural audio tail,
and rejects changed intervals on applied audio-driven takes, which need new generation.
For a matching take already on the timeline, **Reuse selected timeline take** links it to an
approved shot while retaining its footage, prompt and versions. Applying the remaining plan
preserves an existing song region covering the same source interval instead of doubling the mix.
The editing canvas shows the complete music and title extent even while video coverage is short;
this does not extend the exported movie beyond its actual video clips.
Assign first/last images, approve the shots and choose **Add approved selection to timeline**.
The ordinary clip controls then prepare and render them. The original song is placed continuously
on the Music track and generated clip sound is muted. Additional timed frame inputs remain in the
clip inspector. Reference images belonging to objects remain available for endpoint generation
and compatible reference adapters; reference-only adapters are not silently applied to A2V.

Compatible LTX 2.5 continuation shots can form native continuous scenes. Disabling the continuity
preference keeps them independent. Scene groups remain limited to two through six members and
30 seconds. Arbitrary editorial lengths render enough native frames and retain an exact timeline
trim; a fractional final audio interval is preserved with less than one video frame of tail hold.
Source-content verification rejects a changed song before generation. Byte-identical collected or
relinked media remain reusable.

A local two-shot A2V qualification used two image anchors, 384×256 at 24 FPS, and a four-second
interval of a YuE2 song. Generation took 43.8 seconds and a checkpoint resume took 3.4 seconds on
the tested M3 Ultra. Final stereo PCM32 samples matched the original interval exactly and native
AVFoundation playback was verified. This establishes that input combination and source retention;
it does not guarantee lip sync, beat-following movement or seamless results for every scene.

### Native audio analysis qualification

The analysis engine independently implements wav2vec2-base-960h, Beat This small0 and optional
UMX-HQ vocal isolation using MLX; it does not import third-party inference packages. Model setup verifies pinned file hashes and
keeps the Apache-2.0 model card and MIT license alongside the optional downloaded weights.
The stages run sequentially, release their weights and support cancellation. Beat evidence,
estimated stereo vocals and recognition evidence have independent source/model/settings caches;
lyric changes reuse those stages. Corrupted caches are recomputed.

Numerical checks matched the speech reference's frame-token decisions exactly and the beat
reference logits within 0.000023. A 160-second song took 2.12 seconds for combined analysis and
0.18 seconds from cache on the tested M3 Ultra. A known spoken sentence aligned all 16 supplied
words; absent words and lines stayed untimed. These measurements are local qualification, not
a guarantee of accuracy on other sources.

Vocal-isolation numerical checks matched the reference network within relative L2 error 0.00000034.
On the full 159.8-second song, isolation increased supported supplied words from 29/174 to 61/174;
neither mode fully timed a line. End-to-end analysis took 2.09s mixed and 4.38s with isolation,
with 988 MB peak MLX allocation dominated by the sequential acoustic stage on the test machine.
These counts compare acoustic evidence with the exact song's generation-request lyrics, not an
independent transcript or hand-timed boundaries. A qualification-only singing-trained checkpoint
reached 83/174 words after isolation and remains outside the shipped model selection.

Sung words, omission/repetition flags and section labels still require review; acoustic-support
scores are not calibrated probabilities. The optional vocal estimate uses six-second crossfaded
windows and mixture phase, without multistem Wiener refinement. No multilingual aligner, speaker
diarization or guaranteed lip-sync system is included.

### Resumable movie production

Use **Movie → Produce movie…**, or **Shot List → Add approved plan and produce…** after reviewing
the shots and their production objects. The Shot List action adds unapplied approved shots with
their song intervals. Existing timeline clips remain part of the movie.

During production, the shot list refreshes automatically while rendering continues. Pausing
retains completed takes; a delayed status response cannot overwrite the finished run or a
different movie.

Production preparation and headless job export revalidate existing native takes when their
process-local validation is missing, including after reopening Studio. Unchanged resolved inputs
reuse the saved take; changed inputs or missing media still queue generation. An edit during
revalidation stops preparation so a stale result cannot choose what to render.

**Prepare production** captures the edit and execution settings. **Start production** runs native
H3/LTX jobs serially, prepares continuity-dependent shots after their predecessors finish, and
assembles the movie using the existing finishing and audio mixer. Complete LTX2.5 scenes render
as one unit. Scene frame constraints must preserve the reviewed shot lengths; incompatible timing
stops before generation rather than moving music cuts. Source Music regions, stereo channels,
volume/fades, track gain/pan/reverb, titles and track mutes remain part of the final assembly.
Solo affects preview only for new projects; legacy projects retain their export-solo policy.

Choose zero to three automatic retries for local renderer failures. Pause retains completed takes
and scene checkpoints; Resume verifies their hashes before reuse. Cancellation and invalid inputs
are not retried. Assembly publication is journaled so an interruption after the final rename can
resume. The saved project links to its production directory; save the project to retain that link.
Changed source files, runtime settings or global references require a new snapshot, preserving old
outputs. Prepare and render use the same native adapters as individual Studio clips.

Self-hosted Draw Things execution is opt-in and has no automatic resubmission after an uncertain
remote attempt. Cloud API shots still use their individual generation and cost-confirmation flow;
accept those takes before producing the movie. Existing supported model/task restrictions apply.

When assembly completes, preview the movie and use **Apply generated takes** to update the timeline.
Application verifies source/artifact receipts and the unchanged edit/execution context, preserves
older takes and shared scene versions, and leaves production objects and original song data intact.
The original timeline is not replaced during background generation.

Qualification includes real two-clip stereo assembly (48 frames at 24 FPS), distinct left/right test
tones, sample-timing probes, interrupted publication recovery, bounded failure retries, both
cancellation exception types, continuity preparation order and changed-source/changed-context
rejection. The final mixer compensates limiter lookahead so it does not shift the soundtrack.

On an M3 Ultra, a cold two-shot LTX 2.5 audio-driven scene with two image anchors (384×256,
96 frames at 24 FPS) completed generation and assembly in 57.88 seconds. Its verified completed
queue resumed in 0.022 seconds. Native scene PCM matched all 192,000 original stereo samples;
after the mixer correction, the final AAC export had zero measured sample offset on both channels.
These are one local qualification run, not a general throughput guarantee. Rendering quality and
musical gesture/lip synchronization remain model-dependent.
