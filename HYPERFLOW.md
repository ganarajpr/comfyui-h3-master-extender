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
