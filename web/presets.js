// Engine presets for the panel's Preset dropdown. One entry per preset: add a row here and it appears.
// `values` are real widget values; the dropdown itself is never serialized, only these widgets are.
const PIKA = "h3\\pika_h3_flex_ref2va_comfy_bf16.safetensors";
const DMAD = "h3\\minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors";

const REDIVERSE = "h3\\rediverse_pika_h3_flex_ft_comfyui.safetensors";

const PIH3_REFINE = "pi-H3 refine (DWD, no VAE decode)";
// The Standard pass-1 values; the pi-H3 refine presets differ only in the pass-2 engine (the pass-2 LoRA widgets are ignored there).
const PIKA_PASS1 = {
    accel_mode: "Turbo LoRA",
    turbo_lora: PIKA,
    turbo_lora_strength: 1,
    diversity_lora: REDIVERSE,
    diversity_lora_strength: 1,
    diversity_steps: 1,
    pdd_nfe: "6",
    turbo_sampler: "euler",
    turbo_scheduler: "simple",
    pass1_resolution: "608x352 (16:9)",
    pass2_resolution: "1280x720",
};

export const MASTER_PRESETS = [
    {
        id: "standard",
        label: "Standard · Pika + DMAD",
        hint: "Pika H3 flex ref2va LoRA, 6 steps, euler / simple; DMAD refiner, replace mode, 1 refine step (pass2_steps 4 x denoise 0.25). About 65-71 s per 15 s 720p clip.",
        values: {
            accel_mode: "Turbo LoRA",
            turbo_lora: PIKA,
            turbo_lora_strength: 1,
            diversity_lora: REDIVERSE,
            diversity_lora_strength: 1,
            diversity_steps: 1,
            pdd_nfe: "6",
            turbo_sampler: "euler",
            turbo_scheduler: "simple",
            pass2_lora: DMAD,
            pass2_lora_mode: "replace engine LoRA",
            pass2_steps: 4,
            pass2_denoise: 0.25,
            pass1_resolution: "608x352 (16:9)",
            pass2_resolution: "1280x720",
            pass2_chunk_frames: 124,
            pass2_engine: "LoRA refine",
        },
    },
    {
        id: "pika_pih3_refine_2",
        label: "Pika + pi-H3 refine 2-step",
        hint: "Pika pass 1 as Standard; pass 2 is the last step of the pi-H3 2-step DWD grid, and the DWD head's pixels replace the video VAE decode. The chained 2 x 15 s standard test measured 169.9 s.",
        values: {
            ...PIKA_PASS1,
            pass2_engine: PIH3_REFINE,
            pih3_refine_steps: 2,
            pih3_refine_tail: 1,
        },
    },
    {
        id: "pika_pih3_refine_4",
        label: "Pika + pi-H3 refine 4-step",
        hint: "Pika pass 1 as Standard; pass 2 is the last 2 steps of the pi-H3 4-step DWD grid (starts at sigma 0.8993, close to the DMAD refine strength), pixels from the DWD head, no video VAE decode.",
        values: {
            ...PIKA_PASS1,
            pass2_engine: PIH3_REFINE,
            pih3_refine_steps: 4,
            pih3_refine_tail: 2,
        },
    },
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
