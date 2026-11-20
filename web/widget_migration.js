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

// Legacy HyperFlow 8-step (retired as a choice 2026-11-17, kept so workflows saved before re-render as saved). The eight
// widgets hyperflow_file ... pass1_steps are now APPENDED after tst_tau (so every workflow saved since keeps its widget
// order); an older workflow carries them in the middle of the array, which shifts every widget between.
export const HYPERFLOW_WIDGETS = ["hyperflow_file", "hyperflow_curve_refit", "hyperflow_strength", "hyperflow_lora_mode"];
export const DUAL_WIDGETS = ["dual_sampling", "dual_split_step", "pass1_early_stop", "pass1_steps"];
export const LEGACY_HYPERFLOW_WIDGETS = [...HYPERFLOW_WIDGETS, ...DUAL_WIDGETS];

// The four hyperflow_* widgets (file, curve refit, strength, LoRA mode) sat between rewrite_previous_clips and
// audio_refine_steps. A workflow saved before 2026-11-17 carries four extra values there, shifting every later widget.
// The old block is recognised by its types (string, boolean, number, "bypass" / "merge") where the current layout has
// audio_refine_steps (an integer).
// Returns {values, legacy}: the array with the block removed and the removed values by widget name, or null when the
// array has no such block.
export function takeHyperflowValues(names, values) {
    if (!Array.isArray(values)) return null;
    const anchor = names.indexOf("rewrite_previous_clips");
    if (anchor < 0 || names[anchor + 1] !== "audio_refine_steps") return null;
    const block = values.slice(anchor + 1, anchor + 5);
    const [file, refit, strength, mode] = block;
    if (typeof file !== "string" || typeof refit !== "boolean" || typeof strength !== "number") return null;
    if (mode !== "bypass" && mode !== "merge") return null;
    return {
        values: values.filter((_, i) => i < anchor + 1 || i > anchor + 4),
        legacy: Object.fromEntries(HYPERFLOW_WIDGETS.map((name, i) => [name, block[i]])),
    };
}
export function dropHyperflowValues(names, values) {
    return takeHyperflowValues(names, values)?.values ?? null;
}

// dual_sampling, dual_split_step, pass1_early_stop and pass1_steps sat between sparse_sink and pass2_chunk_when. Saved
// arrays carry (boolean, integer, boolean) there, plus an integer pass1_steps in workflows saved after that was added;
// the current layout has pass2_chunk_when's string (or master_ui's "") in that slot. A 3-value block means pass1_steps = 0.
// Returns {values, legacy} like takeHyperflowValues, or null when the array has no such block.
export function takeDualSamplingValues(names, values) {
    if (!Array.isArray(values)) return null;
    const anchor = names.indexOf("sparse_sink");
    if (anchor < 0 || names[anchor + 1] !== "pass2_chunk_when") return null;
    if (typeof values[anchor + 1] !== "boolean" || !Number.isInteger(values[anchor + 2]) || typeof values[anchor + 3] !== "boolean") return null;
    const count = Number.isInteger(values[anchor + 4]) ? 4 : 3;
    const block = values.slice(anchor + 1, anchor + 1 + count);
    return {
        values: values.filter((_, i) => i < anchor + 1 || i > anchor + count),
        legacy: { dual_sampling: block[0], dual_split_step: block[1], pass1_early_stop: block[2], pass1_steps: count === 4 ? block[3] : 0 },
    };
}
export function dropDualSamplingValues(names, values) {
    return takeDualSamplingValues(names, values)?.values ?? null;
}

export const HYPERFLOW_MODE = "HyperFlow 8-step";
export const DUAL_MODE = "HyperFlow 8-step \u00b7 dual sampling";
export const isHyperflowMode = (value) => String(value ?? "") === HYPERFLOW_MODE || String(value ?? "") === DUAL_MODE;
export const isLegacyAccelMode = (value) => String(value ?? "").toLowerCase().startsWith("pdd");

// accel_mode lost its "PDD 8-step" option; a saved workflow that still has it runs HyperFlow 8-step. HyperFlow 8-step
// (plain and dual sampling) is kept as a legacy engine and is not touched.
// Returns the corrected values, or null when there is nothing to change.
export function migrateAccelMode(names, values) {
    if (!Array.isArray(values)) return null;
    const i = names.indexOf("accel_mode");
    if (i < 0 || !isLegacyAccelMode(values[i])) return null;
    const out = [...values];
    out[i] = HYPERFLOW_MODE;
    return out;
}

// All the positional re-seating of a saved UI workflow's widgets_values, earliest slot first (each step reads the array the
// earlier ones left behind): the merged caption model, the legacy HyperFlow block, single_text_encode, the dual sampling block,
// then accel_mode. The legacy HyperFlow / dual values leave the array and come back by widget name in `legacy`: those
// widgets are appended after tst_tau now. Returns {values, legacy}, or null when the array is already in the current layout.
export function migrateSavedValues(names, values) {
    const legacy = {};
    let migrated = dropCaptionModelValue(names, values);
    const steps = [
        (n, v) => { const t = takeHyperflowValues(n, v); if (t) Object.assign(legacy, t.legacy); return t?.values ?? null; },
        dropSingleTextEncodeValue,
        (n, v) => { const t = takeDualSamplingValues(n, v); if (t) Object.assign(legacy, t.legacy); return t?.values ?? null; },
        migrateAccelMode,
    ];
    for (const step of steps) migrated = step(names, migrated ?? values) ?? migrated;
    return migrated ? { values: migrated, legacy } : null;
}

// A workflow saved before TST existed has no value for tst_tau: its values stop at or before the tst_enabled slot (which holds
// master_ui's "" or nothing). It rendered without TST, so it loads with TST off. `values` is the array AFTER migrateSavedValues
// (the legacy HyperFlow values are not in it), so an old HyperFlow workflow is still recognised as shorter than tst_tau's slot.
export function tstOffForSavedValues(names, values) {
    const idx = names.indexOf("tst_tau");
    return Array.isArray(values) && idx >= 0 && values.length <= idx;
}

// chain_memory (after pass1_steps) defaults to "picture + context" on a NEW node. A workflow saved before it existed has no value in
// its slot (or master_ui's ""); it was rendered without chain memory, so it loads with it off and its cached clips stay valid.
export function chainMemoryOffForSavedValues(names, values) {
    const idx = names.indexOf("chain_memory");
    return Array.isArray(values) && idx >= 0 && values.length <= idx;
}

// The eight legacy widgets are appended after tst_tau: a workflow saved since 2026-11-17 has master_ui's "" (or nothing) in
// their slots, which means the defaults. Same as the old repair, so a Turbo / Flash workflow carries the default values.
export const DUAL_DEFAULTS = { dual_sampling: false, dual_split_step: 2, pass1_early_stop: false, pass1_steps: 0 };
export function dualWidgetValue(name, value) {
    const fallback = DUAL_DEFAULTS[name];
    if (typeof fallback === "boolean") return typeof value === "boolean" ? value : fallback;
    if (name === "pass1_steps") return Number.isInteger(value) && value >= 0 && value <= 8 ? value : fallback;
    return Number.isInteger(value) && value >= 1 && value <= 7 ? value : fallback;
}
export function pass1StepsValue(value) {
    return dualWidgetValue("pass1_steps", value);
}
// hyperflow_file keeps any non-empty string (a build that is not installed must fail loudly on the server, not be swapped
// silently); "" / non-string = the default build. The other three: a wrong type = the default.
export const HYPERFLOW_DEFAULTS = { hyperflow_curve_refit: true, hyperflow_strength: 1.0, hyperflow_lora_mode: "bypass" };
export function hyperflowWidgetValue(name, value, fallbackFile) {
    if (name === "hyperflow_file") return typeof value === "string" && value !== "" ? value : fallbackFile;
    const fallback = HYPERFLOW_DEFAULTS[name];
    if (typeof fallback === "boolean") return typeof value === "boolean" ? value : fallback;
    if (name === "hyperflow_lora_mode") return value === "merge" ? "merge" : "bypass";
    return typeof value === "number" && Number.isFinite(value) ? value : fallback;
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
// Anything that is not one of the combo labels = the default.
export const HANDOFF_STEP_OPTIONS = ["1 \u00b7 Pika to 0.98", "2 \u00b7 Pika to 0.96", "3 \u00b7 Pika to 0.92", "4 \u00b7 Pika to 0.86", "5 \u00b7 Pika to 0.71"];
export const HANDOFF_STEP_DEFAULT = HANDOFF_STEP_OPTIONS[2];
export function handoffStepValue(value) {
    return HANDOFF_STEP_OPTIONS.includes(value) ? value : HANDOFF_STEP_DEFAULT;
}

// pass1_cfg, cfg_steps, negative_prompt and cfg_handoff_steps are appended after handoff_step: a workflow saved before them carries
// master_ui's "" (or nothing) in those slots. A non-number falls back to the default; the text reads an empty value as empty.
export const PASS1_CFG_DEFAULT = 4.5;
export const CFG_STEPS_DEFAULT = 24;
export const CFG_HANDOFF_STEPS_DEFAULT = 8;
export function pass1CfgValue(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 1 ? value : PASS1_CFG_DEFAULT;
}
export function cfgStepsValue(value) {
    return Number.isInteger(value) && value >= 2 ? value : CFG_STEPS_DEFAULT;
}
export function cfgHandoffStepsValue(value) {
    return Number.isInteger(value) && value >= 1 ? value : CFG_HANDOFF_STEPS_DEFAULT;
}
export function negativePromptValue(value) {
    return typeof value === "string" ? value : "";
}
