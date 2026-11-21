# HyperFlow Flash (GGUF) mode

> **HyperFlow 8-step was removed on 2026-11-17.** It was the old default acceleration mode (Video Rebirth's 8-step LoRA with two-time `(t, r)` conditioning, applied through the ComfyUI-HyperFlow-H3 pack's `ApplyHyperFlowH3`), together with its dual sampling variants and the inputs `hyperflow_file`, `hyperflow_curve_refit`, `hyperflow_strength`, `hyperflow_lora_mode`, `dual_sampling`, `dual_split_step`, `pass1_early_stop` and `pass1_steps`. The founder decided that it loses to the Pika + DMAD recipe on both speed and quality, so the default recipe is now Pika (`turbo_lora`, 6 steps, euler / simple) with the DMAD refiner on pass 2, and the ComfyUI-HyperFlow-H3 pack is no longer used or needed.
>
> Saved workflows and API graphs with `accel_mode` "HyperFlow 8-step" (or "HyperFlow 8-step · dual sampling", or the older "PDD 8-step") still load and run as Pika + DMAD: Turbo LoRA mode, filling only what the old workflow left unset (`turbo_lora` none -> `h3\pika_h3_flex_ref2va_comfy_bf16.safetensors` at 1.0 with `pdd_nfe` 6, euler, simple; `pass2_lora` none -> `h3\minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors` in replace mode, `pass2_steps` 0 -> 4). The log says the mode was migrated. Stale `hyperflow_*` and dual sampling keys in API graphs are ignored, and the widget panel drops the removed values from saved UI workflows so the later widgets keep theirs. This page now covers only HyperFlow Flash, which is a different engine (a separate compressed transformer, not a LoRA) and is unaffected.

`accel_mode = "HyperFlow Flash (GGUF)"` runs pass 1 on Video Rebirth's HyperFlow H3 Flash: a separate compressed transformer (13.9B ref2va), pruned and low-rank factored, with a two-time (t, r) embedder, shipped as GGUF or INT8 convrot safetensors. It is not a LoRA on the base model.

## Requirements

- Node packs in `custom_nodes`: [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) and [ComfyUI-HyperFlow-Flash](https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash). ComfyUI-GGUF is only needed for GGUF builds. The extender finds the pack through the `HyperFlowLoader` node class (v1.1.0+; `HyperFlowGGUFLoader` on v1.0.0) and loads through its `load_hyperflow` (v1.0.0: `load_hyperflow_gguf`); without the pack the job stops with a message naming it.
- Weights in `diffusion_models`, either kind (the extender renders Ref2VA): `minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors` (the default when installed) or `minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf` (the default otherwise). The file list shows the pack's `safetensors_files()` (only files carrying HyperFlow metadata) plus the GGUFs. Picking a `.safetensors` build with ComfyUI-HyperFlow-Flash older than 1.1.0 stops the job with a message to update the pack. An fl2va pick is swapped for the ref2va sibling with the same quant and extension.
- The accel_mode value stays "HyperFlow Flash (GGUF)" so saved workflows keep working; it now covers both kinds.

## What the mode does

- Pass 1: the Flash transformer, euler, CFG 1, the checkpoint's own 4 / 6 / 8-step grid (`hyperflow_flash_steps`, default 6; the embedded 6-step grid is used for 6). SigmaShift is applied to it like to the base model. The running schedule reaches the two-time embedder through the stock sampler (`transformer_options["sample_sigmas"]`), because pass 1 samples with `comfy.samplers.CFGGuider`.
- The `model` input (base `unet_name` and any LoRAs stacked on it) is **not** used for pass 1; the log says so. It is used for pass 2 and audio refine. `turbo_lora` and `pdd_nfe` are ignored.
- Pass 2: unchanged, the refiner LoRA (DMAD) on the base in replace mode, 1 step at `pass2_denoise` 0.25 with `pass2_steps` 4. With no pass-2 LoRA the plain base would run the tail of the Flash grid; the log warns, pick DMAD.
- Memory: the Flash model is evicted from the GPU right after pass 1 (only it; the VAEs and a background decode are left alone), so it and the base are never resident together. The loaded GGUF stays cached in RAM across clips and jobs (single entry, replaced when another file is chosen). The base is reloaded for pass 2 and Flash for the next pass 1, once per clip.
- File choice: an `fl2va` build is swapped for the `ref2va` file of the same quant when installed, otherwise the job stops (the extender renders Ref2VA).

## Features that change in this mode

| Feature | In Flash mode |
|---|---|
| Feed-forward chunking (KJNodes `MiniMaxChunkFeedForward`) | Skipped for pass 1: it replaces `mlp.forward` assuming the dense fused fc1/fc2, which Flash swaps for low-rank pairs. Pass 2 on the base still uses it. |
| Sparse attention (`sla_enabled`) | Stays on if you set it. Core's block-sparse path calls `attn.qkv_proj(x)` and `attn.out_proj(x)` as modules, so the Flash `SplitQKV` / `LowRankLinear` replacements work. Not validated on Flash weights, and `auto` resolves to the defaults. Turn it off for a dense reference. |
| Motion-context chaining, semantic bridge, audio refine | Unchanged: they act on the conditioning and on the base model, not on the pass-1 transformer. Motion context with Flash is untested. |
| Pass-1 cache | Keyed on the Flash file and steps in addition to the usual settings; other modes' fingerprints are unchanged. |

## Pika x Flash merge

`minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot_pikafold.safetensors` is a Flash checkpoint that must be used with its companion LoRA `h3\pika_h3_flex_ref2va_for_flash.safetensors` at strength 1.0. The extender's own LoRA stack (the model input) only reaches pass 2 in this mode, so nothing could patch the Flash pass-1 model; `hyperflow_flash_lora` / `hyperflow_flash_lora_strength` apply a LoRA to that model (model-only) right after it is loaded. Pick the `_for_flash` LoRA at 1.0 with a pikafold checkpoint; the extender does not select it for you and logs a warning when a pikafold file runs with `none`. The pick is part of the pass-1 cache key.

## Pika → Flash hand-off

`accel_mode` = "Pika → Flash hand-off" runs pass 1 as two models on one noise schedule.

**Why.** Plain Flash gets anatomy wrong where plain Pika does not. Anatomy, pose, composition and prompt compliance are decided in the first, high-noise steps; materials and texture in the last, low-noise ones. So Pika takes the early steps and Flash the late ones: Pika's anatomy and Flash's texture, without merging weights. Neither model is modified.

**How.** Example with `pdd_nfe` 6 and `handoff_step` 3. Pika's schedule is `[1.0, 0.98, 0.9, 0.7, 0.4, 0.15, 0]` (illustrative values):

- Pika samples steps 1..3 on `[1.0, 0.98, 0.9, 0.7]` (its sampler / scheduler, Turbo LoRA mode as usual) and stops at sigma 0.7.
- Flash continues on `[0.7, <its grid points below 0.7>, 0]`, euler, CFG 1, with zero added noise: the latent is already at sigma 0.7. With the 6-step Flash grid that is `[0.7, 0.6969, 0.4688, 0.0]`.
- Same conditioning (references, motion context, bridge) for both segments, both the video and the audio part of the packed latent.
- Pika is evicted before Flash runs, and Flash after pass 1; pass 2 (base + DMAD refine) is unchanged.

**Inputs.** Pika: `turbo_lora` (required, not none), `turbo_lora_strength`, `pdd_nfe`, `turbo_sampler`, `turbo_scheduler`. Flash: `hyperflow_flash_file`, `hyperflow_flash_steps` (only sets the grid Flash continues on), `hyperflow_flash_lora(_strength)`. `handoff_step` (dropdown "1 · Pika to 0.98" ... "5 · Pika to 0.71", default 3; the decimal is the noise level left, exact for pdd_nfe 6): Pika runs steps 1..N of its `pdd_nfe` schedule; it is clamped to `pdd_nfe - 1` (logged). Larger = more Pika (safer anatomy, less Flash texture); smaller = more Flash.

**Cache.** The pass-1 fingerprint includes the Flash parts and `handoff_step` in this mode only, so different hand-offs never share a cached pass 1.

The log line reads `Pika -> Flash hand-off: Pika steps 1..3 (sigma 1.0->0.7000), Flash 3 steps (0.7000->0) [file]`.

# pi-H3 (FL2VA adapter) mode

`accel_mode` = "pi-H3 (FL2VA adapter)" runs pass 1 on [Lakonik/ComfyUI-piFlow](https://github.com/Lakonik/ComfyUI-piFlow)'s pi-H3: a 2- or 4-step adapter that distills MiniMax H3 FL2VA. Pass 2 is unchanged (base model + the pass-2 refiner LoRA, DMAD, replace mode).

**Requirements.** The ComfyUI-piFlow node pack (`Load LakonLab Adapter`, `pi-Flow Scheduler`), `models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors`, and `models/loras/pi_h3_2step_dwd_adapter_int8_convrot.safetensors` / `pi_h3_4step_dwd_adapter_int8_convrot.safetensors`. A missing pack stops the job with a message naming it.

**Inputs** (appended after `chain_memory`, before the `time_gate_json` socket): `pih3_base_file` (default `minimax_h3_fl2va_pruned_int8_convrot.safetensors`), `pih3_steps` (2 or 4, default 2; picks the adapter). `pdd_nfe`, `turbo_lora`, `turbo_sampler` and `turbo_scheduler` do not apply to pass 1 (and are left out of the pass-1 cache key in this mode).

**Pass 1.** The FL2VA base is loaded and the adapter patched in (as the vendor graph: UNETLoader -> Load LakonLab Adapter, strength 1), then SigmaShift (12 / 3), the dense attention backend and, when `sla_enabled`, sparse attention, exactly as for the Flash model. Sigmas come from the pack's `pi-Flow Scheduler` on that model's sampling: `min_raw_t` 0, `max_raw_t` 0.93 (2 steps) / 0.96 (4 steps), `final_step_size_scale` 0.6. Euler, CFG 1, the same noise seed and the same Ref2VA conditioning (references, motion context, bridge) as the other engines, even though the adapter was trained on FL2VA. The patched model is cached across clips and jobs (clip 2 reuses it; the adapter is patched once), and it is evicted from the GPU before pass 2.

**The DWD head is dropped.** The `*_dwd_adapter` files carry `dwd_*` tensors: a pixel head that, at the last sampling step, decodes the final hidden states straight to frames for the `DWD Decode` node. The extender needs a normal video+audio latent (pass 2 refines it and chaining reads its motion context), so `dwd_*` tensors are removed before patching. This changes nothing in the latent: in the piFlow source the head only reads the final hidden states and the model's own output is untouched, so the sampler returns the same latent with or without it. With the head loaded, every sampling run would also stash its frames in `DWD_PIXELS` (up to 4 runs, fp16, a few GB per 15 s clip) and SHA-1 the output latent. The adapter's own denoiser weights were trained with the head; whether they give as good a VAE-decoded latent as the separate non-DWD adapters (`pi_h3_*step_adapter`) is judged on a render.

**Cache.** The pass-1 fingerprint gains a `pih3` entry `[base file, steps]` in this mode only.

