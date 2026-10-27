# comfyui-h3-master-extender

Multi-clip MiniMax H3 video generation for ComfyUI. It runs a two-pass render (HyperFlow 8-step or Turbo LoRA draft, learned 3D latent upscale, refine pass), carries motion and audio continuity from clip to clip, and adds clip validation, project save/load, a built-in prompt rewriter and two story engines that plan a whole film from a story.

This is a community integration, not an official MiniMax or ComfyUI release.

## Origin

This repository started as a fork of [only2uuuu-hub/ComfyUI-MiniMax-H3-Master-Extender-Custom-built-with-Astra-6-](https://github.com/only2uuuu-hub/ComfyUI-MiniMax-H3-Master-Extender-Custom-built-with-Astra-6-), the "Custom built with Astra 6" Master Extender published on 2026-09-11. Thank you to only2uuuu-hub for that build: the two-pass Master Extender node, the Final Decode node, the blank workflow and the installation guide all come from it. Its three commits (`9219cce`, `086b4a4`, `e72acce`) are the base of this repository's history.

Since then the code has diverged a long way (about 150 commits), so on 2026-10-23 the repository was detached from the fork network and renamed `comfyui-h3-master-extender`. The node names, the node folder name and saved workflows are unchanged. The original build itself adapted earlier projects; everyone is credited under [Credits and license](#credits-and-license).

## Features

- **Two-pass render:** a draft pass, a learned 3D latent upscale and a refine pass. HyperFlow 8-step is the default draft engine, with Turbo LoRA as the alternative. A panel preset dropdown fills the engine settings in one click.
- **Clips:** clip-by-clip review or full-batch generation, with per-clip prompts, seeds, durations (5-30 s) and LoRAs. Clips can be validated, re-rolled or rewritten, and every render and cache state shows on the clip card.
- **References:** up to nine reference pictures, plus video and audio references. A clip can use a subset of the pictures (per-clip refs).
- **Continuity:** motion context between clips, run in RAM, and continuation seams decoded from the tail of the previous clip.
- **Caching:** a per-project disk cache. A pass-1 cache survives pass-2 changes, so changing the output size, the refiner or the decode does not re-render the draft. There is also a pass-1-only mode for fast drafts.
- **Prompt rewriter:** a built-in rewriter on a local model that sees the reference pictures. It has a story planner (`auto_clips`) and continuity between clips.
- **Story engines:** `builder`, or the frozen E4.10 / E4.11 story-to-film engines, which plan the film, bind your pictures to its characters and write every clip's final prompt.
- **H3 Story Writer node:** turns a short brief into a story with exactly N beats for the story engine.
- **Decoding:** Final Decode encodes with H.264 NVENC when available and falls back to the CPU. An optional X2 Detail VAE decodes at 2x the refine size.
- **Prompt From Video node:** reads the clip prompts back out of a rendered MP4 or PNG.
- **Speed and memory:** sparse attention (SLA / sol-attn / VSA) with per-pass settings, and VRAM released between passes and clips.

The included workflows contain no prompts, references or saved project content beyond small samples. No model weights, generated media or cache files are included.

## Installation

1. Use a ComfyUI installation with native MiniMax H3 support. The original build was prepared against **ComfyUI 0.34.5** (core commit `7fd919f0caff66a52289ea5b19cb6eaca0da04ef`). Current development runs on a ComfyUI master build from late September 2026. Neither is a claim of testing on every hardware configuration.
2. Open a terminal in `ComfyUI/custom_nodes` and run:

   ```sh
   git clone https://github.com/ganarajpr/comfyui-h3-master-extender.git ComfyUI_MiniMax_H3_Master_Extender
   git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git
   ```

   Keep the folder name `ComfyUI_MiniMax_H3_Master_Extender`. If you installed from the old fork URL, point the existing checkout at this repository instead of cloning a second copy:

   ```sh
   git -C ComfyUI_MiniMax_H3_Master_Extender remote set-url origin https://github.com/ganarajpr/comfyui-h3-master-extender.git
   ```

   Alternatively, extract this project's ZIP into `ComfyUI/custom_nodes/ComfyUI_MiniMax_H3_Master_Extender`. The `__init__.py` file must be directly inside that folder. Install the dependency repository separately; it is not bundled. If already installed, update the existing copies instead of installing duplicates.

3. Install requirements using **the same Python environment that runs ComfyUI**. From the ComfyUI folder with its environment activated:

   ```sh
   python -m pip install -r custom_nodes/ComfyUI_MiniMax_H3_Master_Extender/requirements.txt
   python -m pip install einops
   ```

   Standard ComfyUI supplies PyTorch, NumPy, Pillow, aiohttp, and safetensors. `einops` is used by the external upscaler. Follow any additional installation instructions in the dependency repositories. Do not replace your ComfyUI PyTorch installation with a generic CPU build.

   **Windows portable:** from the portable installation root:

   ```bat
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI_MiniMax_H3_Master_Extender\requirements.txt
   .\python_embeded\python.exe -m pip install einops
   ```

   **ComfyUI Desktop:** use the installation's environment terminal and its actual custom-nodes directory. Do not install dependencies into an unrelated system Python.

4. Download the models separately and place them as described below. Restart ComfyUI, then refresh the browser.
5. Open one of the example workflows, select your installed model files, write a prompt, and attach reference pictures if needed:
   - `example_workflows/MiniMax_H3_Master_Extender_Blank.json` — the HyperFlow 8-step graph with every input at its default.
   - `example_workflows/MiniMax_H3_Master_Extender_Turbo_SLA.json` — the faster graph: Turbo LoRA (ref2v turbo 4-step), comfy kitchen attention with SLA 0.9, chunked refine pass, 608x352 draft to 1280x720, one blank 15 s clip. On an RTX 5090 a 15 s clip renders in roughly 85–90 s. Needs a converted Turbo LoRA in `models/loras` (see `turbo_lora`).
   - `example_workflows/MiniMax_H3_Master_Extender_HyperFlow_Action.json` — the locked action preset: HyperFlow 8-step (pruned file, curve refit on, LoRA mode bypass) with an empty LoRA stack (any LoRA disables the curve fit), SLA 0.9, and the taomate 3-step LoRA replacing the engine LoRA on pass 2. Base model: Singularity v1.3 int8.
   - `example_workflows/MiniMax_H3_Master_Extender_Story_Builder.json` — the HyperFlow action preset set up for building a story clip by clip: built-in rewriter on `pending clips`, continuity `final prompts` (clip N continues from clip N-1's final rewritten prompt, written one after another) and a sample `rewrite_story` (the whole film's story; each clip's writer uses it only to pick that clip's beat). Two clips, the second saying "Continue the story from previous."
   - `example_workflows/MiniMax_H3_Master_Extender_Story_Auto.json` — the Story Builder preset with one empty clip, a five-sentence sample `rewrite_story` and `auto_clips` = 4: the first run plans the story into four clips, then rewrites them one after another.
   - `example_workflows/MiniMax_H3_Master_Extender_Story_Writer.json` — Story Auto with the H3 Story Writer node feeding `rewrite_story` and `story_engine` = `e4.10` (see [H3 Story Writer](#h3-story-writer-brief-to-story)).
   - `example_workflows/MiniMax_H3_Master_Extender_HyperFlow_DualSampling.json` and `MiniMax_H3_Master_Extender_Action_v2_DualSampling.json` — HyperFlow dual sampling (see [HyperFlow dual sampling](#hyperflow-dual-sampling-opt-in)).

   Start with `clip_by_clip` and one clip. Inspect the preview and validate a clip before continuing the chain.

### Other node packs

| Pack | Needed for |
| --- | --- |
| [ComfyUI-HyperFlow-H3](https://github.com/Adudeguyman/ComfyUI-HyperFlow-H3) | **Required** for the default engine, HyperFlow 8-step. See [HYPERFLOW.md](HYPERFLOW.md). |
| [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | **Required.** Provides `MinimaxH3LatentUpscaler3D`, the upscale between the passes. |
| [MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI) | The built-in prompt rewriter, the story engines and the Story Writer node. |
| [ComfyUI-MiniMaxH3_LatentUpscaler](https://github.com/TripleHeadedMonkey/ComfyUI-MiniMaxH3_LatentUpscaler) | The X2 Detail final decode (`final_decode`). |
| [BUNNY H3 Conditioning Bridge](https://github.com/aa335615543-ux/BUNNY_H3_Conditioning_Bridge) | The optional semantic bridge (below). |

You no longer need ComfyUI-MiniMax-H3-PDD-Acc, because the PDD 8-step mode was removed, and H3 Turbo is not a dependency. The standalone Motion Context pack isn't required either, since the adapted motion-context implementation is included here. Avoid duplicate installations of the same node classes.

**Optional: semantic bridge.** The Master node has `semantic_bridge` / `semantic_bridge_alpha` / `semantic_bridge_match` widgets that run the [BUNNY H3 Conditioning Bridge](https://github.com/aa335615543-ux/BUNNY_H3_Conditioning_Bridge) on the conditioning of both passes (model: [JOKER141/BUNNY_H3_Conditioning_Bridge](https://huggingface.co/JOKER141/BUNNY_H3_Conditioning_Bridge)). Install that node and put the adapter in its `models/` folder or in `models/semantic_bridge`; leave `semantic_bridge` at `none` when it is not installed.

## Built-in prompt rewriter: system prompt, continuity, story planner

Turn on `rewrite_mode` and each clip's text is rewritten into an H3 prompt on a local GGUF (needs the MiniMax-H3-Prompt-Rewriter-ComfyUI pack).

- **One rewriter model, one call per clip** (`rewrite_writer_model`). The same model looks at the reference pictures and writes the prompts; there is no caption step and no separate caption model (old workflows' `rewrite_caption_model` value is dropped on load). A model sees images when the pack lists it as a captioner, i.e. it ships an mmproj (a ninfer artifact brings its own vision tower), or, for *Strata*, when it is serving with its vision section (the gateway config `strata-iq3_s.gateway.json` has `--vision` in `args` and a `vision` entry whose `strata-vision` and mmproj files exist; a Strata already running is asked through `/v1/models`). The planner call and every clip's writer call then carry the reference pictures themselves, on one server session, with thinking available. **Layout:** each picture is downscaled once to at most 896 px on its long side (a multiple of 28, the vision tower's block size: a square is 1,024 tokens, a 3:4 portrait 768, so nine pictures cost at most 9,216 tokens; 1,024 is also Strata's per-picture ceiling) and sent as a PNG data URI. They come FIRST in the user message, as `Picture 1:` + image ... `Picture N:` + image (N up to 9), in the same order and byte for byte in every call; everything that varies per clip (task, duration, the reference list, story, previous clips, the ask) follows them. The system prompt is also identical in every clip's call (its rules are decided per run, not per clip), so a server that caches the prompt prefix (llama-server, ninfer, possibly Strata) reads the pictures once. The log has one line with the picture count, the sent sizes and the estimated image tokens; with Strata a second line gives the prompt tokens the server reported. Reference *videos* are still captioned on a llama/ninfer session (cached by frame hash); Strata labels them without describing them. A model without vision (a GGUF with no mmproj, or a Strata without the vision section) takes the old path: no image input, text-only rewrite, and the log says `writer has no vision: reference images not interpreted`. The `<Picture N>` labels still reach the writer as plain lines (`Picture 1: an attached reference picture (not described here)`), so the Ref2VA structure is unchanged; a rule is added telling the writer not to guess what a picture shows and to take wardrobe from the ask, the previous clip and the story. The planner then gets the story text alone.
- **Thinking budget** (`rewrite_reasoning_budget`): a dropdown of `1024`, `2048`, `4096` (default `4096`), thinking tokens at most, on top of `rewrite_max_new_tokens`. Saved workflows and API graphs that send another number (`-1`, `0`, `8192`, `16384`...) are mapped to the nearest allowed value (`<=1024` to 1024, above 4096 and `-1`/`0` to 4096).
- **System prompt.** `rewrite_system_prompt_in` (socket) wins, then the `rewrite_system_prompt` widget. **When both are empty the default is now the H3 Prompt Studio's builder, `prompts/builder.md`, for Ref2VA** (reference pictures connected). It is written for ref2va (`<Subject N>` / `<Picture N>`, six sections), so a T2VA run (no references) keeps MiniMax's official guide. To get the official guide in every case, type `@official` in the widget (a value in the existing text widget, so saved widget positions do not move). **Workflows saved with an empty system prompt now get the builder from their next rewrite; clips already rewritten are not redone** (the system prompt is not part of a clip's fingerprint).
- **Continuity for clip N** (`rewrite_previous_clips`): `off`; `raw asks` (default: clips 1..N-1 as you typed them, clips written in parallel); `final prompts` (the full final prompt of clip N-1 plus the raw asks of clips 1..N-2; clips are written one after another, and clip N is rewritten again when N-1's final prompt changes).
- **Film story** (`rewrite_story`): the whole story, given to every clip's writer as a `story:` block so it covers only its own beat. Does not mark rewritten clips stale.
- **Story planner** (`auto_clips`, default 0 = off). With a story set and `auto_clips` = N, the first run plans the story once into exactly N clips of `auto_clip_seconds` each (INT, 5-15, default 15) (the Studio's one-call chapter-breakdown planner, `prompts/planner.md`, on the writer GGUF), writes each clip's shot list into the empty clips as its raw ask (flagged *planned*, editable in the panel), then rewrites them. Clips you typed are kept and planned around: if any clip in the range has text, the planner is shown every clip in order (typed ones verbatim, the empty ones as `[TO PLAN: clip N]`) and plans only the empty ones in one call, each bridging the clip before it to the clip after it (clips after the last typed one simply continue it); with nothing typed it plans the whole story into N. Typed clips are never overwritten. Changing the story or N never plans again (that would discard rendered takes); use **Replan from story** in the Prompt Rewriter section. A run queued from the API without the panel does not store the plan, so it plans again next time. **Plan more:** raise `auto_clips` past the number of clips and only the new ones are planned, numbered after the existing ones, continuing from where the last one ends (the planner is given the whole story, the existing clips' asks in order — your edited text if you edited one — and the state carried out of the last clip). Existing clips, typed or planned, are never touched; lowering `auto_clips` or editing the story does nothing; an empty clip inside the existing range stays empty. With `final prompts` continuity, the first new clip continues from the last existing clip's final prompt. **Reference pictures** (`planner_refs`, default `images`): the planner is shown the connected pictures, each labelled `Picture N:`, with a rule to stage the story only with them and to name a subject by its Picture number the first time it appears in a clip. `images` sends the real pictures (the same ones, in the same order, as the clip writers get) and needs a rewriter model with vision (see above); `captions` is the old name for `images` and now means the same. With a rewriter model that cannot see images the planner gets the story text alone. `off` gives the planner the story text alone. Reference videos are caption lines only. Each picture is budgeted at (width//28)*(height//28) context tokens after the 896 px downscale (at most 1,024). The rule is in code, not in `prompts/planner.md`.

`prompts/builder.md` and `prompts/planner.md` are copies of the Studio's `DEFAULT_REWRITE_SYSTEM_PROMPT` and `CHAPTER_BREAKDOWN_TEMPLATE`; the canonical copy lives here, and the Studio has a test that fails when they drift.

## Story engine: `story_engine` = `builder` | `e4.10` | `e4.11`

Story mode (`rewrite_story` + `auto_clips`) has two engines. **`builder`** (the default) is everything described above: `planner.md` plans the story, then `builder.md` writes every clip from the reference pictures. **`e4.10`** plans the whole film and writes every clip's FINAL prompt with the E4.10 story-to-film pipeline; `builder.md` is not run for those clips.

**What E4 does that the builder does not.** Every spoken word is written verbatim into the plan, including counting and repeated calls, and checked across clips; every speaker has one id (`(S1)`, `(S2)`) for the whole film; a voice nobody sees is written as an off-screen voice, never as a `<Subject>`, and its cut either shows no face or says the visible character's lips are pressed shut; `non_diegetic_music` is `N/A` unless you ask for a score, and hum/drone/resonance words are swept out of the sound design; one film bible fixes who and what exists, with an acting profile per character; one small decision call per clip chooses who is on screen, the staging and the references, and a ledger carries state (positions, wardrobe, props) from clip to clip, and a visible physical change the plan declares (a tear, a torn sleeve, an oil sheen on a hull) is carried into the subject line and the retention line of every later clip until the plan resets it (E4.8), the film bible describes each such subject in its opening state only and the base line of a subject that carries a later value holds none of that axis's opening wording, so `dry cheeks, clear eyes` never sits next to `now with wet tear trails` (E4.9, E4.10); the writer's draft is checked by code (lines verbatim, speakers, cut markers, durations, negations, spatial words) and repaired where it fails. The extender's `final prompts` continuity is therefore not needed for these clips. Frozen at `e4.10-frozen`; the engine is measured and described in the eval repo, not here.

**Settings** (all appended last, so saved workflows keep their widget positions; old workflows load with the defaults, which are the builder):

| widget | values | meaning |
| --- | --- | --- |
| `story_engine` | `builder` (default), `e4.10`, `e4.11` | the engine of the first plan. `e4` (a workflow saved with the E4.6/E4.7 build) and `e4.8` (the E4.8 build) are read as `e4.10`. `e4.11` is E4.11.1: the same pipeline built for few repair calls (fixes in code first) and parallel requests (up to `rewrite_parallel`, at most 4, in flight at once); the first run on the box is its proof, no E4.11.1 run is replayed in the tests |
| `e4_language` | text, default `English` | the language every spoken line is written in |
| `e4_score` | `off` (default), `on` | `off`: `non_diegetic_music: N/A`. `on`: the film bible writes one instrumental score that every clip carries |
| `e4_decision_budget` | `off` (default), `1024`, `2048`, `4096` | thinking of each per-clip decision call (who is on screen, staging, tactic). `off`: no thinking. Measured on 1,937 decision questions with Swift 1.5: off is 0.6 points below thinking 2048 (p = 0.48, not significant) and about 4.7 times faster (2.9 s against 13.8 s per call); 1024 is worse than both. The planner, the film bible, the writer and the repairs always think up to 4096, and so does the picture-binding call (2048: it looks at images, and with thinking off it missed a character that thinking 2048 and 4096 found) |
| `e4_picture_notes` | text, one line per picture | `2: the old tailor in a grey kurta`. A hint beside the picture for a model that sees; the only way to bind pictures for a model that cannot |

**How a run goes.** With `story_engine` = `e4.10`, `rewrite_mode` on, a `rewrite_story`, `auto_clips` of 1 or more, an empty clip list and at least one connected reference picture, the first run does this instead of the planner and the per-clip writer calls (the panel's progress line says each step):
1. The rewriter's model server is opened as always (a ninfer artifact, a llama.cpp model, or Strata), with thinking on and a context sized for E4 (65536 tokens per slot, `MINIMAX_H3_E4_SLOT_CTX` to change it; 32768 cut a long Hindi plan). Each call's `max_tokens` is its own setting lowered to the slot minus the estimated prompt minus a margin, so a short prompt keeps nearly the whole slot; `MINIMAX_H3_E4_MAX_TOKENS` is an optional hard ceiling. A reply the server stops at `max_tokens` is sent once more with a larger cap (never the identical request), the planner and bible retries ask for shorter field values, and the failure names the cut (`reply cut at max_tokens N (thinking T)`). A long story (more than about 14 clips) is planned in parts of about 8 clips that carry the film so far, and a part that cannot be planned is split smaller (`e4/bridge/scale.mjs`). If E4.10 still fails, the node does not stop: it warns loudly (status line and note, naming the cause and the kept run folder) and plans with the builder planner.
2. E4 runs as ONE Node subprocess against that same server: planner, film bible, **picture binding**, one decision call per clip, one writer call per clip (up to 4 at once, at most `rewrite_parallel`), repairs only where a check fails. Thinking is on for the planner, the bible, the writer and the repairs (4096 tokens at most, a real per-request cap) and, by default, OFF for the per-clip decision calls (the picture-binding call thinks 2048).
3. Its clips are written into the clip list as finished clips; the rest of the run (motion context, rendering) is unchanged.

**The model is the rewriter's model; nothing else is called.** `e4_engine.endpoint_of` reads the server the rewriter opened. A ninfer artifact is called on `/v1/messages` with `thinking.budget_tokens` (a real per-request cap), and `"thinking": {"type": "disabled"}` where thinking is off (the field is never omitted: omitting it means the model thinks). A llama.cpp server or Strata is called on `/v1/chat/completions` with the top-level fields `reasoning_budget_tokens` and `reasoning_budget_message` where thinking is on, and `chat_template_kwargs: {"enable_thinking": false}` where it is off (a top-level `enable_thinking` returns an empty message); `max_tokens` is capped to what a slot holds. The writer must run on a server: a vision model from the pack's list, or Strata (a plain GGUF that runs in-process cannot be called by a subprocess, and the run stops with that message).

**Pictures.** E4 does not render reference images: yours are bound to its characters, props and locations. After the film bible, one decision call shows the model every connected picture (the same 896 px PNG data URIs, `Picture N:` first, that the extender sends everywhere; N is the slot number) with the list of entities, and asks, for each picture, which ONE entity it shows or `unused`. The reply is checked (every picture answered, every id real, an entity gets one picture; one retry with the complaint) and settled in code. The clips then cite `<Subject k> is NAME in <Picture j>, ...` with YOUR Picture numbers, not renumbered per clip, so the connected pictures match the prompts; an entity with no picture stays "described in words only" and cites none; a voice is never a subject. A model without vision binds by `e4_picture_notes`; without notes (or when its reply is unusable) a plain match of the note lines to entity names is the last resort, and without either every entity is described in words only (the log says so). The result is `story/pictures/map.json` in the run directory.

**Example (E4.10, the stored Swift 1.5 run of `drama_two_hander` replayed through the extender; `tests/fixtures/e4_replay.json.gz`).** Five pictures connected: 1 a plate no entity shows, 2 Mohan, 3 Tarun, 4 the boat, 5 the ghat. The model's answers (planner, bible, decisions, writers, repairs) are the ones the lane gave in that run (13 requests, 0 misses); only the picture-binding call is answered by the test's own mapping, because that run was made by the eval repo's runner and did not bind pictures. The plan declares one visible trace, the oil sheen on the boat, set by clip 1. Clip 1's boat line is the picture's own, and clips 2 and 3 carry the change:
```
clip 1: <Subject 3> is Mohan's wooden boat in <Picture 4>, a narrow wooden rowing boat with a dark varnished finish; the object's own material, scale and form, held from its anchor plate.
clip 3: <Subject 3> is Mohan's wooden boat in <Picture 4>, a narrow wooden rowing boat with a dark varnished finish, now with a fresh oil sheen along the hull; the object's own material, scale and form, held from its anchor plate.
        <Subject 3>: partially_preserved - identity and material, scale and form from its reference picture; a fresh oil sheen along the hull from the previous clip
```
The numbers are the user's: the subjects are numbered per clip, the pictures are not. The base line says nothing of the hull's opening state ("a dry hull with no oil sheen"): the bible is told not to write it, and the line assembly strips it if it does. E4.10 on the tears story, clip 2: `<Subject 1> is Marisol, a young woman with a narrow frame and a dark cotton shirt, now with wet tear trails down both cheeks, eyes red-rimmed; ...` (E4.9 had `with dry cheeks, clear eyes, ..., now with wet tear trails`).

**What the clips look like.** Each planned clip becomes an ordinary finished clip: `prompt` is the six-section prompt, `prompt_raw` the planned ask (shots, camera, action, lines), `prompt_rewritten` and `planned` true, `title` `Clip N: <beat>`, `duration` a whole number of seconds (E4 plans on the 17k+5 frame grid, 15.08 s for a 15 s clip, which becomes 15), and `rewrite_meta` carries `engine: "e4"`, the engine version, the picture list, any citation issue and the extender's own fingerprint, so a later run does not rewrite them again. A clip E4 could not write keeps its planned ask as a pending clip and the builder writes it in the normal pass (a note says so).

**Requirements.**
- **Node.js 18 or newer** (plain executable; E4 has no npm package and needs no install). The extender looks for it in this order: the environment variable `MINIMAX_H3_E4_NODE`, the first line of `<ComfyUI>/user/minimax_h3_master/e4_node.txt`, then `node` on PATH. The error names what was tried.
- The `e4/` folder (shipped here). `node e4/verify-frozen.mjs` proves it is the frozen E4.10: every file is hashed against `e4/MANIFEST.json` and against the freeze record `e4/planpath410/E4.10-FROZEN.md`, and the three files that had to be patched (`hybrid3/lib.mjs`: the optional dhee-core `.env`; `planpath410/hybrid46/validate.mjs`: the vendored audit runner; `planpath410/film.mjs`: the picture-binding hook) are verified with their declared replacements (`e4/PATCHES.json`) reversed. With the eval repo, `--eval-repo <path>` also compares with the git tag `e4.10-frozen`.
- A server-backed rewriter, at least one reference picture, and the story in `rewrite_story`.

**What differs from the frozen E4.10.** Nothing in its prompts, checks or logic: the patches listed in `e4/PATCHES.json` (the three above, plus the ones for cut replies and long films: `planpath410/transport.mjs`, `lib.mjs`, `planner.mjs`, `bible.mjs`, `writer.mjs`, `refs.mjs`, and more hooks in `film.mjs`; the long-film logic itself is `e4/bridge/scale.mjs`, outside the frozen set) are the only differences, and `node e4/verify-frozen.mjs` proves it. E4.10 is E4.9 plus three fixes, E4.9 is E4.8 plus three, and E4.8 is E4.7 plus one rule (the eval repo's `planpath410/E4.10-FROZEN.md` and the older records list the changes and the proof). E4.8: a visible state change the plan declares carries into every later clip's subject and retention lines. E4.9: the film bible describes every subject in its opening state only, an option of a trace axis must read as a noun phrase after "now with", and the planner is asked for the aftermath of every beat (tears, sweat, wet or torn clothes, wounds, mud). E4.10: the base line of a subject carrying a later value drops that axis's opening wording (and the bible is asked not to write it), the name rule rejects only the redundant forms (a value that starts with the subject or whose head noun is the subject), and the aftermath check needs an axis of the cue's kind (crying needs tears, eyes or cheeks, not just any axis). E4.7 differed from E4.6 only in its transport: the decision calls think off by default (`E4_DECISION_THINKING=off|<budget>`), the `llama-chat` wire, and `E4_LLM_MAX_TOKENS`. E4.6's film-level measurements were made with thinking 4096 on every call kind, so they do not cover that default. The vendored file list is the freeze's own `closure.mjs` list plus the whole `vendor/prompts`, `vendor/schemas` and `templater/taxonomy` folders (`closure.mjs` misses a prompt opened through a built path, which the replay test caught).

**What the first version supports, and what it does not.**
- It plans a whole film into an EMPTY clip list: the first plan, or the first plan after "Replan from story" (which empties the planned clips as before and starts E4 again, a new sample).
- E4 decides the number of clips from the story and plans 15 s clips. `auto_clips` only switches planning on (a different number is logged, not used) and `auto_clip_seconds` is not used (the planner is the frozen one; other lengths were never measured).
- Typed clips and "Plan more" (a raised `auto_clips` after a plan) are not supported by E4: that run uses the builder planner, and the panel says so. The film's other clips stay as they are.
- Clips are written into positions 1..N; empty clips beyond N are left empty (a note says how many). Every connected picture is attached to every clip's render, as for the builder; a clip's prompt cites only the pictures whose entity is on screen in it.
- "Rewrite again" on one clip uses `builder.md` on the planned ask, not E4. To redo the film with E4, use "Replan from story".
- The planner works from the story text; it does not see the pictures (it plans before the bible exists). Pictures bind to the entities the story names.
- `rewrite_previous_clips` (raw asks / final prompts) is not used for E4's clips: E4's ledger carries continuity. It still applies to clips the builder writes.

**Logs.** `<ComfyUI>/user/minimax_h3_master/e4/<run>/`: `job.json`, `calls.jsonl` (one line per model call), `story/` (`plan.json`, `bible.json`, `pictures/map.json`, per call `*.request.json` / `*.response.raw.txt` / `*.meta.json`, per clip `decisions.json`, `facts.txt`, `final.prose.json`, checks), `clips.json` (what was put into the clip list) and `prompts/`. Pictures are logged by hash, not as base64. A run that stops early resumes from its stored plan and bible when the same story is run again.

**Tests.** `python -m unittest discover -s tests -p "test_e4_engine.py"` (Node, numpy and Pillow; no GPU, no ComfyUI) replays a stored real E4.10 run through a mock model server and the real Node subprocess (and, with no pictures, reproduces that run's plan, bible, state fold, shots and each clip's facts, writer prompt and final prose byte for byte), and replays four story-mode scenarios to prove the builder's requests are byte for byte those of commit b12b03d; `node --test e4/bridge/test/bridge.test.mjs`; `tests/e2e/e4_outside_comfyui.py` runs one story against a real model with the extender's own code (ComfyUI and the rewriter pack stubbed). `DEPLOY-E4.md` has the steps to put this on the box.

## H3 Story Writer (brief to story)

A separate node, `H3 Story Writer (brief to story)` (category *MiniMax H3 Master*). The Master Extender node is not changed by it. Wire `[H3 Story Writer] --story--> [Master Extender story input]` with `story_engine` = `e4.10`: E4 treats its story input as a finished story (one clip per distinct dramatic beat, every spoken line copied verbatim), so a three-line brief gives about four clips with invented dialogue. The writer turns the brief into a story with exactly N beats, so E4 plans about N clips.

- **Inputs:** `brief`, `clips` (1-60, default 8), `seconds` (per clip, default 15; sizes each beat and the speech in it, at most about 2 words a second), `language` (same meaning as `e4_language`; a language not written in Latin script is written in its own script), `seed` (part of the resume key), `writer_model` (the rewriter pack's list, as `rewrite_writer_model`: a vision model on ninfer-serve or llama-server, or Strata) and `thinking_budget` (1024 / 2048 / 4096, default 4096). **Output:** `story` (STRING).
- **Format it writes:** plain prose, one paragraph per beat, a blank line between beats, exactly N beats. Each beat is one filmable moment: who is where, what they physically do, what visibly changes. Every spoken line is in quotes with the speaker named in the same sentence. A character is described once, in full, where they first appear; physical state (tears, wet clothes, a removed garment, a wound) is stated and carried. No headings, labels or markdown. All characters are adults; a brief that puts a minor in a sexual context raises an error and nothing is written.
- **How:** an OUTLINE (cast, locations and exactly N one-line beats, each with an `end_state`: place, who is present, visible state of clothes, body and props), then WRITE calls of 4 to 8 beats (sized so a story goes out in about one wave: 40 beats on 8 slots is 5 a chunk, on 6 slots 7) **run in parallel**, `concurrency` at a time (optional widget, default 6, max 8; set it to the number of slots the server was started with). A chunk never reads another chunk's prose: it is given the brief, the cast sheet, the locations, the whole outline with every beat's end state, the beats before and after its own and the opening state of its first beat. The outline reply is parsed tolerantly (alternate keys, beats as objects, broken quoting salvaged by key), so an "unusable" reply is now rare (it was a straight double quote inside a spoken line, or a beats list under another key or in another shape). A wrong beat count is repaired, not asked for again in full: a few too many are merged in code, more by one call that returns only the merges, too few by splitting the longest beats in parallel. From 31 beats the outline is acts first (one call), then the beats of every act in parallel. After writing, cheap code checks look at every seam (an opening in a place or with a person the previous end state rules out, a name that is in no cast sheet) and report them as warnings and in the sidecar (`seam_issues`); only a wrong beat count, an unparseable reply or the minor sentinel causes a retry. A cut reply keeps its complete beats and retries the rest with a larger cap and a terser instruction; a call that still fails is halved. Every accepted beat is stored at once under `output/h3_story_writer/_work/<key>/beats/`; running the same inputs again resumes. Per-call `max_tokens` is the slot context (`MINIMAX_H3_E4_SLOT_CTX`, default 65536) minus the estimated prompt, as in E4; every call carries the thinking budget (`/v1/messages` `thinking.budget_tokens` on ninfer, `reasoning_budget_tokens` plus the budget message on llama.cpp and Strata).
- **Model server:** the same one E4 uses inside ComfyUI (the rewriter pack's `mtmd.session` / Strata), one slot (parallel requests queue until the server is started with more slots; the node does not change that), started for this node and closed when it ends, followed by the extender's VRAM release, so the render that follows has the card. An unreachable server raises an error; the brief is never passed through as the story.
- **Files:** `output/h3_story_writer/<timestamp>_<n>clips.md` (the story) and `.json` (brief hash, clips requested, beats written, model, token counts, stop reasons, warnings, call log); the path is printed in the console.
- **Tests:** `python -m unittest discover -s tests -p "test_story_writer.py"` (no model, no ComfyUI). `tests/e2e/story_writer_outside_comfyui.py --brief-file brief.txt --clips 6 --out DIR --e4` runs the writer against the 5090 gateway's ninfer lane and then plans the story with the vendored E4.10 (clips requested vs planned, utterances, lines verbatim).
- **Workflow:** `example_workflows/MiniMax_H3_Master_Extender_Story_Writer.json` is the Story Auto example with the writer wired into `rewrite_story`, `story_engine` set to `e4.10` and a core *Preview as Text* node on the story. `python tools/add_story_writer.py in.json out.json [--clips 8] [--seconds 15] [--no-preview]` does the same to any saved UI-format workflow with a Master Extender node (the writer goes to the left of the graph, takes the Extender's story text as its brief and its writer model and language; running it again changes nothing). Tests: `tests/test_add_story_writer.py`.
- **To deploy:** copy `story_writer.py`, `prompts/story_writer_*.md` and the changed `__init__.py` into the extender folder (or pull the branch), then restart ComfyUI so the node appears. Nothing else changes.

## Presets

The panel's Engine section starts with a **preset** dropdown that fills the engine widgets in one click. It only writes ordinary widgets (so saved workflows and API graphs are unaffected) and shows **Custom** whenever the current values match no preset.

| Preset | Engine | Pass 1 | Pass 2 | 15 s clip |
|---|---|---|---|---|
| Standard | HyperFlow 8-step, DMAD refiner (4 steps, replace) | 608x352 | 1280x720 | ~122 s |
| K4 / K5 / K6 / K7 · 352p | HyperFlow dual sampling, split 4/5/6/7, early stop | 608x352 | 1280x720 | ~198 / ~176 / n.m. / ~112 s |
| K4 / K5 / K6 / K7 · 480p | same | 864x480 | 1280x720 | ~188 / ~172 / ~158 / n.m. s |

Add a preset by adding one entry to `MASTER_PRESETS` in `web/presets.js`.

## HyperFlow 8-step mode

The default acceleration mode (the other is Turbo LoRA). See [HYPERFLOW.md](HYPERFLOW.md) for the node pack, weights, base/build pairing, the pass-2 rule and the sparse attention recipe inputs (`sparse_start_percent`, `sparse_dense_blocks`, `sparse_sink`).

## Render controls

All of these inputs are appended after the existing ones, so saved workflows keep their widget positions. A workflow that does not send an input renders as before, unless noted. [CHANGELOG.md](CHANGELOG.md) has the full detail for each.

| Input / feature | What it does |
| --- | --- |
| `pass2_resolution` = `off (pass 1 only)` | Skips pass 2 and decodes the draft at the pass-1 size (2x that with the X2 decode). Its text encode runs at the 720p size, so a draft matches the pass 1 of a 720p render. |
| Pass-1 cache | The draft chain on disk is keyed only on pass-1 inputs. Changing the pass-2 resolution, the pass-2 LoRA or steps, `final_decode`, or turning pass 2 on or off reuses the cached pass 1 (`Clip N: PASS 1 LOADED FROM CACHE`). The log says why when it can't. |
| `pass1_steps` | Default 0, the full 8 steps. k = 1..7 stops HyperFlow pass 1 after k steps and hands the clean x0 estimate to the normal upscale and refine. It is ignored with dual sampling or Turbo LoRA. |
| Per-clip refs | In `clips_json`, a clip with `"refs": [i, j, ...]` (1-based indices into the job's pictures) is conditioned only on those pictures, numbered `<Picture 1..k>` in that order. A clip without it gets every picture. A bad list fails before any clip renders. The prompt rewriter still writes against all job pictures. |
| `final_decode` | `standard`, or a 2x VAE such as MiniMax-H3-X2-Detail, which decodes the final video at 2x the refine size (about +23 s per 15 s clip). Only final clips use it, and the VAE is found by its safetensors header. |
| `pass2_sla_sparsity` | Default 0, which means the same as `sla_sparsity`. Otherwise it sets a pass-2-only keep rate for the sla / vsa methods (0.95 keeps 5% of key blocks) while pass 1 keeps `sla_sparsity`. sol-attn ignores it. |
| `pass2_chunk_when` | `above 720p` (default) or `always`. See [HyperFlow dual sampling](#hyperflow-dual-sampling-opt-in). This one does change the default at 720p: those clips are now refined in one window. |
| VRAM release | No input. Node-scoped VRAM is released between passes and between clips. ComfyUI's "unload all models" also frees the extender's cached X2 VAE and empties the CUDA cache, so an external model server is not starved after a job. |
| MiniMax H3 Prompt From Video | A separate node. It reads the prompt graph a render embedded in its MP4/PNG and returns the Master Extender clip prompts as STRING outputs that plug into `clip_prompt_1..4`. |

## PTD sampling (opt-in, A/B)

`accel_mode = PTD (adapter)` runs pass 1 with a PTD adapter (per-hop LoRA + PDD heads, one model evaluation per trained hop) from
[ComfyUI-MiniMaxH3-PTD](https://github.com/dheeai/comfyui-minimaxh3-ptd) (which needs
[ComfyUI-PTD](https://github.com/vita-epfl/PTD-inference)). Pick the file with `ptd_adapter` (`models/ptd`). The pack's loader, controller
and sampler do all the PTD numerics; the extender only wires them in under its attention backend and SLA.

| arm | `accel_mode` | `ptd_adapter` | `pass2_engine` | pass 2 |
|---|---|---|---|---|
| A (existing) | `Turbo LoRA` | `none` | `LoRA refine` | `pass2_lora` refine |
| B | `PTD (adapter)` | the file | `LoRA refine` | unchanged refine; needs `pass2_lora` (replace mode is forced) |
| C | `PTD (adapter)` | the file | `PTD last hop` | the adapter's final hop only, at pass-2 size, on the re-noised upscaled draft (one evaluation; `pass2_lora`, `pass2_steps`, `pass2_denoise` unused) |

Notes: a 6-hop file has grid 0.999, 0.9826, 0.9590, 0.9222, 0.8563, 0.7052, 0, so arm C re-noises to sigma 0.7052. Arm C refines the whole
clip in one pass (a windowed pass 2, i.e. `above 720p` output over 800 px short side, is refused). Adapter factors live in host RAM and are
uploaded per layer per forward; nothing adapter-sized stays on the GPU. The final-clip disk cache fingerprint does not include the model or
accel settings, so clear the cache by hand when switching arms.

## HyperFlow dual sampling (opt-in)

Three inputs at the end of the node, all off by default (a workflow or API graph that does not send them renders exactly as before):

| input | default | meaning |
|---|---|---|
| `dual_sampling` | off | HyperFlow 8-step only. Pass 2 = the tail of HyperFlow's trained grid `[1.0, 0.9939, 0.9843, 0.9661, 0.9231, 0.8349, 0.6969, 0.4688, 0.0]` from index `dual_split_step`. No pass-2 LoRA (`pass2_lora` is ignored, with a warning) and `pass2_denoise` is ignored. |
| `dual_split_step` | 2 | Grid index where pass 2 takes over: 2 = 6 pass-2 steps from sigma 0.9843, 6 = 2 steps. |
| `pass1_early_stop` | off | Experimental, needs dual sampling. Pass 1 runs only `grid[:k+1]`. |

The image is generated at the pass-2 size and pass 1 only sets the layout. Recommended: pass 1 `608x352` (renders matched a 1056x608 pass 1 and are cheaper), pass 2 `1344x768`, split 2 (see `example_workflows/MiniMax_H3_Master_Extender_HyperFlow_DualSampling.json`). In Turbo LoRA mode dual sampling logs a warning and runs the normal two passes.

**Early stop.** Pass 1 runs the first k steps, then the model's clean (x0) estimate from the last step is upscaled, and pass 2 re-noises it to `grid[k]` the same way every pass 2 does (the sampler blends noise into the start latent at the first sigma, same seed) and runs the rest. The 3D latent upscaler was trained on clean latents, which is why the x0 estimate is used instead of the noisy step latent, but quality is not validated; keep it off for approved looks. With pass 2 off, early stop is ignored.

**Caching.** `dual_sampling` alone does not change pass 1, so pass-1 caches keep hitting. Early stop adds `dual: [k, true]` to the pass-1 fingerprint, only when on. As with any other pass-2 setting, the final-clip cache does not track these inputs: re-roll the clip (or change the seed) to re-render pass 2.

### Engine mode "HyperFlow 8-step · dual sampling"

The same thing as one `accel_mode` choice, appended last so saved workflows and API graphs are untouched (the default stays "HyperFlow 8-step"). Picking it means: HyperFlow 8-step, dual sampling on with split = `dual_split_step` (default 2), no pass-2 refine LoRA, and pass 2 over the whole clip in one window (`pass2_chunk_frames` is treated as 0 in this mode, which avoids window seams for a near-full regeneration; whole-clip pass 2 at 1344x768 x 362 frames ran in VRAM at about 33.5 s/step). The `dual_sampling` toggle keeps working in the other modes. The panel shows the HyperFlow rows plus split step and early stop, and the badge reads "HyperFlow · dual sampling". Pass-1 caches are shared with plain HyperFlow (the mode resolves to it before the fingerprint is built).

**Pass-2 windowing (`pass2_chunk_when`, appended last).** `above 720p` (default): pass 2 is refined in temporal windows (`pass2_chunk_frames`/`pass2_chunk_overlap`) only when its short side is over 800 px (1080p-class). 720p-class output (1280x720/736, 1344x768) refines the whole clip in one pass, which is faster with sparse attention since the window overlaps are not recomputed. `always` windows any clip longer than `pass2_chunk_frames`; `pass2_chunk_frames` 0 never windows.

Example: `example_workflows/MiniMax_H3_Master_Extender_Action_v2_DualSampling.json` (Singularity ref2va v2 pruned int8, pass 1 `608x352`, pass 2 `1344x768`, pruned HyperFlow file with curve refit, SLA 0.9, early stop off). **Use prompts under 5,000 characters.**

## Debugging: per-step x0 dump

Off unless a flag file exists (no node inputs, no widgets; with the file absent the render path is unchanged). It answers "at which pass-1 step does the clip's structure and motion stop changing?", to choose a pass-1 to pass-2 handoff step.

**Enable.** Create an empty file `<ComfyUI base dir>/user/master_x0_debug`; it is checked on every pass-1 sample call, so no restart is needed. **Delete it afterwards**: while it exists every pass 1 writes its latents and decodes frames, which costs disk and time.

**Output.** `<ComfyUI output dir>/master_x0_debug/<prompt_id or timestamp>/` (a further clip in the same run goes to `<id>_2`, `<id>_3`):

- `step_<NN>_sigma_<s>.pt`: the video part of the model's x0 after step NN (1-based; sigma is the step's input sigma), mapped through `process_latent_out` like the early-stop x0. A dict with `x0`, `step`, `sigma`, `sigma_next`.
- `step_<NN>_frame_<F>.png`: pixel frames F at 10/50/90% of the clip, decoded from that x0 with the engine's video VAE after sampling ends (whole-latent decode, pass-1 size).
- `pass1_final.pt` (the pass-1 output latent's video part) and `meta.json`.

It runs on normal pass 1 as well as early stop. Every part is wrapped so a dump failure is logged (`x0 debug dump:` lines, one per step) and never breaks the render.

**Analyse** (ComfyUI's embedded python, CPU is fine; needs numpy, torch and Pillow):

    python tools/x0_step_analysis.py <dump_dir> [--json out.json] [--sheet out.png]

Prints a table per step, writes `analysis.json` and `contact_sheet.png` (rows = steps, columns = decoded frames). Distances are relative L2 against the last step (`/fin`) or the previous step (`/prev`): `struct` on a low-pass x0 (average pool 4 spatially, 2 temporally), `detail` on the residual x0 minus low-pass, `motion` on frame-to-frame differences of the low-pass. When `struct/fin` and `motion/fin` are near 0 while `detail/fin` is still large, the remaining steps only add detail.

## Model files (not included)

These are the filenames selected in the example workflows. Choose compatible alternatives in the dropdowns if you use another supported precision or filename; renaming an incompatible model does not make it compatible.

| Purpose | Folder under `ComfyUI/models` | Example workflow selection |
| --- | --- | --- |
| Ref2VA diffusion model | `diffusion_models` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` (the action presets use `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8` / `_v2_pruned_int8`) |
| Qwen3-VL text encoder | `text_encoders` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| Video VAE | `vae` | `minimax_h3_video_vae_int8_convrot.safetensors` |
| Audio VAE | `vae` | `minimax_h3_audio_vae_fp32.safetensors` |
| Learned 3D latent upscaler | `latent_upscale_models` | `minimax_h3_latent_upscaler_3d_fp16.safetensors` |
| HyperFlow 8-step | `hyperflow` | `custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors` (pruned bases) or the build without `_pruned` (full base) |
| Pass-2 refiner LoRA (default) | `loras/h3` | `minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors` (optional; with none, pass 2 runs on HyperFlow's own grid tail) |

Keep the CLIPLoader type set to `minimax`. The selected quantized models require compatible ComfyUI kernels and hardware; they are not universal defaults.

- Upscaler weights: [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler).
- For the base model, encoder, and VAEs, use the model provider's MiniMax H3 instructions and compatible ComfyUI builds. This release does not provide or automatically download them.

Model licenses are separate from the source-code license. Hardware requirements depend on model precision, resolution, clip duration, and offloading. No minimum VRAM guarantee is made.

## ComfyUI launch arguments

No project-specific command-line flag is required. From the ComfyUI folder with its Python environment activated, a simple starting command is:

```sh
python main.py --use-pytorch-cross-attention --preview-method none
```

Windows portable, from the portable installation root:

```bat
.\python_embeded\python.exe -s .\ComfyUI\main.py --windows-standalone-build --use-pytorch-cross-attention --preview-method none
```

For additional memory headroom, you can try:

```sh
python main.py --use-pytorch-cross-attention --preview-method none --reserve-vram 2 --cache-none
```

| Argument | Effect |
| --- | --- |
| `--use-pytorch-cross-attention` | Selects core PyTorch attention globally. |
| `--preview-method none` | Disables core sampler previews; the custom video preview still works. |
| `--reserve-vram 2` | Reserves 2 GB of VRAM for other use; adjust for your machine. |
| `--cache-none` | Disables core node-result caching, reducing its memory use at the expense of re-execution. It does not delete this extension's disk cache. |
| `--cpu-vae` | Optional CPU VAE fallback, generally slower and still requiring system RAM. |
| `--port 8188` | Chooses the server port; 8188 is the default. |

The **Master node's `attention_backend` selection controls both sampling passes independently of the global attention flag**. The blank workflow selects `pytorch attention`. Choose `comfy kitchen attention` or `sage attention 2.2` only when the corresponding backend is installed and supported. Keep ComfyUI's normal precision/offloading defaults unless your hardware needs a specific adjustment. In the inspected core, `--lowvram` has no effect when dynamic VRAM is enabled.

For Desktop, put the desired flags in your existing launch configuration if supported, or use its environment terminal. Check `python main.py --help` for the arguments supported by your installed version.

## Outputs and troubleshooting

- The extension writes working data (cache chains) into `ComfyUI/output/MasterExtender_cache/`; keep it writable and allow sufficient disk space. A `cache/` folder left by older versions inside the node folder is moved there automatically on first use. Caches are generated locally and are not part of the release.
- Leave Final Decode's output directory empty to use the default ComfyUI output location. Use its preview/export controls or the connected SaveVideo node to save a result.
- Missing `MinimaxH3LatentUpscaler3D`: install the upscaler dependency and its model.
- Missing native H3 modules or attention API errors: check the ComfyUI version and startup import errors.
- FFmpeg not found: install this project's requirements into ComfyUI's Python, or provide FFmpeg on PATH. H.264 automatically falls back to software encoding if NVENC is unavailable.
- Memory errors: reduce clip duration/resolution, keep smart offload enabled, and process clips individually. Launch flags cannot make every model fit every GPU.
- Refresh the browser after updates. Avoid running old and new copies of this package together.

## Credits and license

This project brings together adapted code and separately installed dependencies. Credit does not imply endorsement.

| Original author / project | Contribution |
| --- | --- |
| [only2uuuu-hub / ComfyUI-MiniMax-H3-Master-Extender-Custom-built-with-Astra-6-](https://github.com/only2uuuu-hub/ComfyUI-MiniMax-H3-Master-Extender-Custom-built-with-Astra-6-) | The original Master Extender build (2026-09-11) this repository was forked from: the two-pass Master node, Final Decode, the blank workflow and the installation guide. |
| [tritant / ComfyUI_MiniMax_H3_Extender](https://github.com/tritant/ComfyUI_MiniMax_H3_Extender) | Extender foundation and inherited continuity/decode implementation. |
| [NikoDemon80 / ComfyUI-H3-Motion-Context](https://github.com/NikoDemon80/ComfyUI-H3-Motion-Context) | GPL-3.0 motion-context temporal anchor and payload patch foundation. |
| [LBH-123-AI / Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) | External learned latent upscaling dependency and its weights. The original build linked xmarre's copy, now [xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus). |
| [Adudeguyman / ComfyUI-HyperFlow-H3](https://github.com/Adudeguyman/ComfyUI-HyperFlow-H3) | External HyperFlow 8-step node pack (Video Rebirth's HyperFlow). |
| [pytraveler / MiniMax-H3-Prompt-Rewriter-ComfyUI](https://github.com/pytraveler/MiniMax-H3-Prompt-Rewriter-ComfyUI) | External prompt-rewriter pack the built-in rewriter runs on. |
| [Comfy-Org / ComfyUI](https://github.com/Comfy-Org/ComfyUI) | Runtime and native MiniMax H3 model integration. |

The combined source distribution is provided under **GNU GPL version 3**. Preserve the upstream notices and Apache-2.0 terms for the portions originally provided under Apache-2.0. See [LICENSE](LICENSE), [THIRD_PARTY_NOTICE.md](THIRD_PARTY_NOTICE.md), and [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt). External dependencies retain their own licenses.

**Modifications.** As GPL-3.0 section 5(a) requires, this repository is a modified version of the work above. Ganaraj Permunda has modified it continuously from 2026-09-11, and [CHANGELOG.md](CHANGELOG.md) and the git history record every change.
