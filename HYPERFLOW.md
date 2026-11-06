# HyperFlow 8-step mode

`accel_mode = "HyperFlow 8-step"` is the default acceleration mode (the other is "Turbo LoRA"; the old "PDD 8-step" mode was removed, and a saved workflow or API graph that still says it runs HyperFlow). It runs Video Rebirth's HyperFlow: an 8-step LoRA plus two-time `(t, r)` conditioning.

## Requirements

- Node pack: [ComfyUI-HyperFlow-H3](https://github.com/Adudeguyman/ComfyUI-HyperFlow-H3) (v1.4.0 or later) in `custom_nodes`, no extra Python requirements. The extender calls its `ApplyHyperFlowH3` node class. Restart ComfyUI after installing it.
- Weights in `models/hyperflow/`: `custom_node_hyperflow_8step_v1.0_comfyui.safetensors` (full base) and `custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors` (pruned base).

## Base and build pairing

| Loaded base | `hyperflow_file` | `hyperflow_curve_refit` |
|---|---|---|
| Full MiniMax-H3 (has `time_embedder`) | the build without `_pruned` | ignored |
| Pruned base with a bundled curve fit (e.g. `Minimax-h3_Singularity_ref2va_Pruned_v1.3_int8`) | the `_pruned` build | on (default) |
| Any other pruned base | the `_pruned` build | LoRA-only, single-time (off-recipe); the log says so |

A mismatched pairing stops the job with the node pack's own message. With refit on and a matching base, the ComfyUI log must not say `curve refit disabled`.

## What the mode does

- Pass 1 uses the MODEL and the trained 9-point SIGMAS returned by `ApplyHyperFlowH3` (applied on the SigmaShift model). Never a BasicScheduler schedule. `pdd_nfe` (the Turbo LoRA step count) is ignored (8 fixed), and the sampler is Euler.
- New inputs, appended after every existing widget so saved workflows keep their widget order: `hyperflow_file`, `hyperflow_curve_refit`, `hyperflow_strength`.
- Sparse attention: `sla_enabled` behaves as in the other modes and the log notes what is in effect (method, tau or keep %, start, dense_blocks, sink, and whether each came from the recipe, the default or your setting). HyperFlow's own validated sol-attn recipe is `start_percent 0.16`, `dense_blocks "0,1"`, `tau 1.0`, sink off. New inputs `sparse_start_percent` (-1 = auto), `sparse_dense_blocks` (`auto`) and `sparse_sink` (`auto`), appended last: with HyperFlow 8-step + `sparse_method` sol-attn, auto resolves to start 0.16, dense blocks 0,1, sink off; in every other mode auto is the old fixed 0.20 / none / exact_kv_and_rows. Any value you set wins. `sparse_tau` is not changed automatically (default 1.3; the recipe is 1.0).

## Pass-2 rule

HyperFlow must never run on a sigma tail that is not cut from its own grid.

- No pass-2 LoRA: pass 2 uses the last `round(8 x pass2_denoise)` steps of HyperFlow's 9-point grid (3 points at the default 0.25), on the HyperFlow model.
- Pass-2 LoRA set: the mode is forced to "replace engine LoRA". Pass 2 runs on the SigmaShift model plus that LoRA, without HyperFlow, on a scheduler tail as in Turbo mode. The log says which case applied.

## Curve-refit limits (measured)

The curve fit is bound to the exact checkpoint and the exact recipe, by the node pack itself:

- It needs an unmodified MODEL. Any LoRA applied before the extender (for example a combat LoRA in an `LTX_lora_loader` stack) makes the node log `curve refit disabled: requires an unmodified checkpoint`, and that job runs LoRA-only. An empty or all-off stack keeps the fit.
- It applies only on the full 9-point grid, so the pass-2 tail always runs backbone-only (the node logs `curve refit disabled for an unmatched sampling recipe` once per job). Pass 1 keeps the fit.

## Old workflows

Workflows saved before HyperFlow (UI or API format) load with the three HyperFlow inputs at their defaults. UI workflows that stored the `master_ui` DOM widget's empty string used to push `""` into the `hyperflow_file` slot and fail queueing with "Some input values are not available"; `onConfigure` in `web/master_extender.js` now resets any missing or invalid HyperFlow value to the node-definition default, and the server treats an empty `hyperflow_file` as the default (a non-empty unknown name is still rejected).

## Cache

The disk cache does not re-render when settings change. Clear the chain (panel "clear cache", or `POST /comfyui/minimax_master/clear_cache`) before comparing engines.

## `hyperflow_lora_mode` (optional, default `bypass`)

- `bypass`: the reference behaviour. The rank-256 LoRA is computed on top of the model at every step.
  Measured cost is about 25% more time per pass-1 step than the plain model (3.44 s against 2.74 s per step at 608x352).
- `merge`: the LoRA is folded into the weights once, so each step should cost about the same as the
  plain model. The node pack warns that merging into quantized (int8) weights can change numerical
  results, so A/B it before relying on it. The two small base time projections stay in bypass in both modes.

# HyperFlow Flash (GGUF) mode

`accel_mode = "HyperFlow Flash (GGUF)"` runs pass 1 on Video Rebirth's HyperFlow H3 Flash: a separate compressed transformer (13.9B ref2va), pruned and low-rank factored, with a two-time (t, r) embedder, shipped as GGUF. It is not a LoRA on the base model.

## Requirements

- Node packs in `custom_nodes`: [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) and [ComfyUI-HyperFlow-Flash](https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash). The extender finds the pack through the `HyperFlowGGUFLoader` node class and calls its `load_hyperflow_gguf`; without it the job stops with a message naming the pack.
- Weights in `diffusion_models`: `minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf` (the extender renders Ref2VA).

## What the mode does

- Pass 1: the Flash transformer, euler, CFG 1, the checkpoint's own 4 / 6 / 8-step grid (`hyperflow_flash_steps`, default 6; the embedded 6-step grid is used for 6). SigmaShift is applied to it like to the base model. The running schedule reaches the two-time embedder through the stock sampler (`transformer_options["sample_sigmas"]`), because pass 1 samples with `comfy.samplers.CFGGuider`.
- The `model` input (base `unet_name` and any LoRAs stacked on it) is **not** used for pass 1; the log says so. It is used for pass 2 and audio refine. `turbo_lora` and `pdd_nfe` are ignored.
- Pass 2: unchanged, the refiner LoRA (DMAD) on the base in replace mode, 1 step at `pass2_denoise` 0.25 with `pass2_steps` 4. With no pass-2 LoRA the plain base would run the tail of the Flash grid; the log warns, pick DMAD.
- Memory: the Flash model is evicted from the GPU right after pass 1 (only it; the VAEs and a background decode are left alone), so it and the base are never resident together. The loaded GGUF stays cached in RAM across clips and jobs (single entry, replaced when another file is chosen). The base is reloaded for pass 2 and Flash for the next pass 1, once per clip.
- File choice: an `fl2va` build is swapped for the `ref2va` file of the same quant when installed, otherwise the job stops (the extender renders Ref2VA).

## Features that change in this mode

| Feature | In Flash mode |
|---|---|
| Dual sampling, `pass1_steps`, `hyperflow_file` / curve refit / strength / lora mode | Not applicable (HyperFlow 8-step features). Ignored with a warning. |
| Feed-forward chunking (KJNodes `MiniMaxChunkFeedForward`) | Skipped for pass 1: it replaces `mlp.forward` assuming the dense fused fc1/fc2, which Flash swaps for low-rank pairs. Pass 2 on the base still uses it. |
| Sparse attention (`sla_enabled`) | Stays on if you set it. Core's block-sparse path calls `attn.qkv_proj(x)` and `attn.out_proj(x)` as modules, so the Flash `SplitQKV` / `LowRankLinear` replacements work. Not validated on Flash weights, and the HyperFlow sol-attn auto recipe is not applied (`auto` = the defaults). Turn it off for a dense reference. |
| Motion-context chaining, semantic bridge, audio refine | Unchanged: they act on the conditioning and on the base model, not on the pass-1 transformer. Motion context with Flash is untested. |
| Pass-1 cache | Keyed on the Flash file and steps in addition to the usual settings; other modes' fingerprints are unchanged. |
