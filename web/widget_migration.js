// Saved-workflow migrations for the Master node's rewriter widgets.

const CAPTION_LENGTHS = ["brief", "standard", "detailed"];
const REASONING_BUDGETS = [1024, 2048, 4096];

// rewrite_caption_model sat right after rewrite_writer_model and has been merged into it. A workflow saved before
// that carries one extra value there, which shifts every later widget by one. The old layout is recognised by the
// caption length (a short fixed list) sitting two slots after the writer instead of one.
// Returns the corrected values, or null when the array is already in the current layout.
export function dropCaptionModelValue(names, values) {
    if (!Array.isArray(values)) return null;
    const writer = names.indexOf("rewrite_writer_model");
    if (writer < 0 || names[writer + 1] !== "rewrite_caption_length") return null;
    if (CAPTION_LENGTHS.includes(values[writer + 1]) || !CAPTION_LENGTHS.includes(values[writer + 2])) return null;
    return values.filter((_, i) => i !== writer + 1);
}

// single_text_encode (a BOOLEAN between audio_refine_cache and final_decode) was removed: one text-encoder pass per
// clip is now always on. A workflow saved before that carries one extra boolean there, shifting every later widget.
// Returns the corrected values, or null when the array is already in the current layout.
export function dropSingleTextEncodeValue(names, values) {
    if (!Array.isArray(values)) return null;
    const cache = names.indexOf("audio_refine_cache");
    if (cache < 0 || names[cache + 1] !== "final_decode") return null;
    if (typeof values[cache + 1] !== "boolean" || typeof values[cache + 2] !== "string") return null;
    return values.filter((_, i) => i !== cache + 1);
}

// rewrite_reasoning_budget was an INT (-1..32768) and is now a 1024 / 2048 / 4096 dropdown of strings.
// Same mapping as prompt_rewriter.reasoning_budget: nearest allowed value, -1 / 0 / junk -> 4096.
export function normalizeReasoningBudget(value) {
    const n = Math.trunc(Number(value));
    if (!Number.isFinite(n) || n <= 0) return String(REASONING_BUDGETS[REASONING_BUDGETS.length - 1]);
    let best = REASONING_BUDGETS[0];
    for (const b of REASONING_BUDGETS) if (Math.abs(b - n) <= Math.abs(best - n)) best = b;
    return String(best);
}

// The four hyperflow_* widgets (file, curve refit, strength, LoRA mode) sat between rewrite_previous_clips and
// audio_refine_steps and were removed with HyperFlow 8-step (2026-11-17). A workflow saved before that carries four
// extra values there, shifting every later widget. The old block is recognised by its types (string, boolean, number,
// "bypass" / "merge") where the current layout has audio_refine_steps (an integer).
// Returns the corrected values, or null when the array is already in the current layout.
export function dropHyperflowValues(names, values) {
    if (!Array.isArray(values)) return null;
    const anchor = names.indexOf("rewrite_previous_clips");
    if (anchor < 0 || names[anchor + 1] !== "audio_refine_steps") return null;
    const [file, refit, strength, mode] = values.slice(anchor + 1, anchor + 5);
    if (typeof file !== "string" || typeof refit !== "boolean" || typeof strength !== "number") return null;
    if (mode !== "bypass" && mode !== "merge") return null;
    return values.filter((_, i) => i < anchor + 1 || i > anchor + 4);
}

// dual_sampling, dual_split_step, pass1_early_stop and pass1_steps sat between sparse_sink and pass2_chunk_when and were
// removed with HyperFlow 8-step. Saved arrays carry (boolean, integer, boolean) there, plus an integer pass1_steps in
// workflows saved after that was added; the current layout has pass2_chunk_when's string (or master_ui's "") in that slot.
// Returns the corrected values, or null when the array is already in the current layout.
export function dropDualSamplingValues(names, values) {
    if (!Array.isArray(values)) return null;
    const anchor = names.indexOf("sparse_sink");
    if (anchor < 0 || names[anchor + 1] !== "pass2_chunk_when") return null;
    if (typeof values[anchor + 1] !== "boolean" || !Number.isInteger(values[anchor + 2]) || typeof values[anchor + 3] !== "boolean") return null;
    const count = Number.isInteger(values[anchor + 4]) ? 4 : 3;
    return values.filter((_, i) => i < anchor + 1 || i > anchor + count);
}

// The Pika + DMAD recipe: the founder's default engine since 2026-11-17 (HyperFlow 8-step lost to it on speed and quality).
// Keep in step with master_node.PIKA_LORA / migrate_legacy_accel.
export const PIKA_LORA = "h3\\pika_h3_flex_ref2va_comfy_bf16.safetensors";
export const DMAD_LORA = "h3\\minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors";
export const TURBO_MODE = "Turbo LoRA";
export const isLegacyAccelMode = (value) => {
    const text = String(value ?? "").toLowerCase();
    return text.startsWith("pdd") || text.startsWith("hyperflow 8-step");
};

// accel_mode lost its "PDD 8-step" and "HyperFlow 8-step" (plain and dual sampling) options; a saved workflow that still
// has one runs the Pika + DMAD recipe: Turbo LoRA, and only what the old workflow left unset is filled (the "none"
// turbo LoRA with its 6 steps / euler / simple, the "none" pass-2 LoRA with DMAD, replace mode and 4 steps).
// Returns the corrected values, or null when there is nothing to change.
export function migrateAccelMode(names, values) {
    if (!Array.isArray(values)) return null;
    const accel = names.indexOf("accel_mode");
    if (accel < 0 || !isLegacyAccelMode(values[accel])) return null;
    const out = [...values];
    const get = (name) => out[names.indexOf(name)];
    const set = (name, value) => { const i = names.indexOf(name); if (i >= 0) out[i] = value; };
    const unset = (name) => { const v = get(name); return v === undefined || v === "" || v === "none"; };
    out[accel] = TURBO_MODE;
    if (unset("turbo_lora")) {
        set("turbo_lora", PIKA_LORA);
        set("turbo_lora_strength", 1.0);
        set("pdd_nfe", "6");
        set("turbo_sampler", "euler");
        set("turbo_scheduler", "simple");
    }
    if (unset("pass2_lora")) {
        set("pass2_lora", DMAD_LORA);
        set("pass2_lora_strength", 1.0);
        set("pass2_lora_mode", "replace engine LoRA");
        if (!Number(get("pass2_steps"))) set("pass2_steps", 4);
    }
    return out;
}

// pass2_chunk_when is the first widget after sparse_sink: a workflow saved before it has master_ui's "" there. Anything else invalid = default.
export function pass2ChunkWhenValue(value) {
    return value === "always" ? "always" : "above 720p";
}
// pass2_sla_sparsity is appended after pass2_chunk_when: a workflow saved before it has master_ui's "" there. 0 = same as sla_sparsity.
export function pass2SlaSparsityValue(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0.5 && value <= 0.99 ? value : 0;
}
// pass2_scheduler and pass2_audio are appended after pass2_sla_sparsity: a workflow saved before them has master_ui's "" in
// pass2_scheduler and nothing in pass2_audio. Both mean the defaults (same scheduler as pass 1, pass-2 refined audio).
export const PASS2_SCHEDULER_DEFAULT = "same as pass 1";
export const PASS2_AUDIO_DEFAULT = "pass 2 (refined)";
// A workflow saved on the PTD branch carries ptd_adapter ("none" or an adapter file) in this slot: anything that is not
// one of the widget's own options falls back to the default.
export function pass2SchedulerValue(value, options) {
    if (typeof value !== "string" || value === "") return PASS2_SCHEDULER_DEFAULT;
    if (Array.isArray(options) && !options.includes(value)) return PASS2_SCHEDULER_DEFAULT;
    return value;
}
export function pass2AudioValue(value) {
    return value === "pass 1" ? "pass 1" : PASS2_AUDIO_DEFAULT;
}
// hyperflow_flash_file and hyperflow_flash_steps are the last widgets (after pass2_audio): a workflow saved before them has
// master_ui's "" in the file slot. "" / unknown file = the baked Pika x Flash merge, else the INT8 ref2va build, else the Q8_0 GGUF,
// else the first installed option; steps outside 4 / 6 / 8 = 6.
export const FLASH_FILE_PREFERENCE = [
    "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot_pikamerge.safetensors",
    "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors",
    "minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf",
];
export const FLASH_STEPS_DEFAULT = 6;
export function flashStepsValue(value) {
    return value === 4 || value === 6 || value === 8 ? value : FLASH_STEPS_DEFAULT;
}
export function flashFileValue(value, options) {
    if (typeof value === "string" && value !== "" && (!Array.isArray(options) || options.includes(value))) return value;
    if (Array.isArray(options) && options.length) return FLASH_FILE_PREFERENCE.find(n => options.includes(n)) ?? options[0];
    return FLASH_FILE_PREFERENCE[0];
}
// hyperflow_flash_lora and hyperflow_flash_lora_strength are appended after hyperflow_flash_steps: a workflow saved before them
// carries master_ui's "" in the lora slot. "" / unknown lora = "none"; a non-number strength = 1.
export const FLASH_LORA_STRENGTH_DEFAULT = 1.0;
export function flashLoraValue(value, options) {
    if (typeof value !== "string" || value === "") return "none";
    if (Array.isArray(options) && !options.includes(value)) return "none";
    return value;
}
export function flashLoraStrengthValue(value) {
    return typeof value === "number" && Number.isFinite(value) ? value : FLASH_LORA_STRENGTH_DEFAULT;
}
// handoff_step is appended after hyperflow_flash_lora_strength: a workflow saved before it carries master_ui's "" in that slot.
// A non-integer or out-of-range value = 3.
export const HANDOFF_STEP_DEFAULT = 3;
export function handoffStepValue(value) {
    return Number.isInteger(value) && value >= 1 && value <= 7 ? value : HANDOFF_STEP_DEFAULT;
}
