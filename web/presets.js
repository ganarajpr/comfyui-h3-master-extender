// Engine presets for the panel's Preset dropdown. One entry per preset: add a row here and it appears.
// `values` are real widget values; the dropdown itself is never serialized, only these widgets are.
const PRUNED = "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors";
const DMAD = "h3\\minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors";
const DUAL_MODE = "HyperFlow 8-step · dual sampling";
// pass2_refiner values (pdd_pure_engine.PASS2_REFINERS)
export const REFINE_LORA = "refine lora";
export const OPEN_H3 = "Open-H3 regenerate + face";

const kPreset = (k, res, label, hint) => ({
    id: `k${k}${res === "864x480 (16:9)" ? "-480p" : ""}`,
    label,
    hint,
    values: {
        accel_mode: DUAL_MODE,
        hyperflow_file: PRUNED,
        hyperflow_curve_refit: true,
        dual_sampling: true,
        pass1_early_stop: true,
        dual_split_step: k,
        pass1_resolution: res,
        pass2_resolution: "1280x720",
        pass2_lora: "none",
        pass2_steps: 0,
        pass2_refiner: REFINE_LORA,
    },
});

const STANDARD_VALUES = {
    accel_mode: "HyperFlow 8-step",
    hyperflow_file: PRUNED,
    hyperflow_curve_refit: true,
    pass2_refiner: REFINE_LORA,
    pass2_lora: DMAD,
    pass2_lora_mode: "replace engine LoRA",
    pass2_steps: 4,
    pass2_denoise: 0.25,
    pass1_resolution: "608x352 (16:9)",
    pass2_resolution: "1280x720",
    dual_sampling: false,
    pass1_early_stop: false,
    dual_split_step: 2,
    pass2_chunk_frames: 124,
    pass1_steps: 0,
};

const stdP1Preset = (k) => ({
    id: `standard-p1-${k}`,
    label: `Standard \u00b7 pass 1 = ${k} steps`,
    hint: `The Standard recipe with pass 1 stopped after ${k} of 8 HyperFlow steps; its clean estimate is upscaled and the normal refine runs. ${k === 7 ? "2-ref clip: 87.6 s vs 88.9 s Standard." : k === 6 ? "2-ref clip: 82.2 s vs 88.9 s Standard." : "Not measured."} Not measured on the DEVI fight.`,
    values: { ...STANDARD_VALUES, pass1_steps: k },
});

export const MASTER_PRESETS = [
    {
        id: "standard",
        label: "Standard · ~132 s",
        hint: "HyperFlow 8-step, DMAD refiner. Times in this list are the DEVI fight (15 s, 9 refs) unless noted: 132 s with the whole-clip pass 2 (146 s windowed). Fewer refs are faster (2 refs: ~89 s).",
        values: STANDARD_VALUES,
    },
    {
        // no resolutions here: pass 2 refines to whatever refine resolution is selected, like the other refiners
        id: "open-h3",
        label: "Open-H3 refine + face",
        hint: "HyperFlow 8-step pass 1, then NVIDIA's Open-H3-Regenerate-2K refiner as pass 2 (3 steps from sigma 0.909) with a face map, so faces start at 0.50 and keep their identity. Pass-1 audio is kept. Keeps the selected draft and refine resolutions.",
        values: {
            accel_mode: "HyperFlow 8-step",
            hyperflow_file: PRUNED,
            hyperflow_curve_refit: true,
            pass2_refiner: OPEN_H3,
            dual_sampling: false,
            pass1_early_stop: false,
            pass2_chunk_frames: 124,
            pass1_steps: 0,
        },
    },
    stdP1Preset(7),
    stdP1Preset(6),
    kPreset(4, "608x352 (16:9)", "K4 · 352p · ~198 s", "Dual sampling, split 4, 352p pass 1. DEVI fight 197.5 s."),
    kPreset(5, "608x352 (16:9)", "K5 · 352p · ~176 s", "Dual sampling, split 5, 352p pass 1. DEVI fight 175.6 s."),
    kPreset(6, "608x352 (16:9)", "K6 · 352p · not measured", "Dual sampling, split 6, 352p pass 1. Not measured."),
    kPreset(7, "608x352 (16:9)", "K7 · 352p · not measured", "Dual sampling, split 7, 352p pass 1. Not measured on the DEVI fight; CHOODI c03 (6 refs): 111.9 s vs 122.1 s Standard."),
    kPreset(4, "864x480 (16:9)", "K4 · 480p · ~230 s", "Dual sampling, split 4, 480p pass 1. DEVI fight 230.3 s (CHOODI c21c, 5 refs: 188 s)."),
    kPreset(5, "864x480 (16:9)", "K5 · 480p · ~212 s", "Dual sampling, split 5, 480p pass 1. DEVI fight 211.6 s (CHOODI c21c: 174 s)."),
    kPreset(6, "864x480 (16:9)", "K6 · 480p · ~196 s", "Dual sampling, split 6, 480p pass 1. DEVI fight 195.9 s (CHOODI c21c: 160 s)."),
    kPreset(7, "864x480 (16:9)", "K7 · 480p · not measured", "Dual sampling, split 7, 480p pass 1. Not measured."),
];

export const CUSTOM_PRESET_ID = "custom";

const same = (a, b) => (typeof b === "number" ? Number(a) === b : typeof b === "boolean" ? a === b : String(a) === String(b));

// getValue(name) -> the widget's current value. Returns the first preset whose every listed value matches, else null.
export function matchPreset(getValue, presets = MASTER_PRESETS) {
    for (const p of presets) {
        if (Object.entries(p.values).every(([name, want]) => {
            const have = getValue(name);
            return have !== undefined && have !== null && same(have, want);
        })) return p;
    }
    return null;
}
