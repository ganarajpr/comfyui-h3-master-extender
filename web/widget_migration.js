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

// accel_mode lost its "PDD 8-step" option; a saved workflow that still has it runs HyperFlow 8-step.
// Returns the corrected values, or null when there is nothing to change.
export function migrateAccelMode(names, values) {
    if (!Array.isArray(values)) return null;
    const i = names.indexOf("accel_mode");
    if (i < 0 || !String(values[i] ?? "").toLowerCase().startsWith("pdd")) return null;
    const out = [...values];
    out[i] = "HyperFlow 8-step";
    return out;
}

// dual_sampling, dual_split_step and pass1_early_stop are the last widgets, after sparse_sink. A workflow saved before them
// has its trailing master_ui "" land in dual_sampling, and any missing trailing value stays undefined: both mean the default.
export const DUAL_DEFAULTS = { dual_sampling: false, dual_split_step: 2, pass1_early_stop: false };
// pass1_steps is appended after pass1_early_stop: a workflow saved before it has master_ui's "" there. 0 = full pass 1.
export function pass1StepsValue(value) {
    return Number.isInteger(value) && value >= 0 && value <= 8 ? value : 0;
}
// pass2_chunk_when is appended after pass1_steps: a workflow saved before it has master_ui's "" there. Anything else invalid = default.
export function pass2ChunkWhenValue(value) {
    return value === "always" ? "always" : "above 720p";
}
// pass2_sla_sparsity is appended after pass2_chunk_when: a workflow saved before it has master_ui's "" there. 0 = same as sla_sparsity.
export function pass2SlaSparsityValue(value) {
    return typeof value === "number" && Number.isFinite(value) && value >= 0.5 && value <= 0.99 ? value : 0;
}
// upfront_text_encode is appended after pass2_sla_sparsity: a workflow saved before it has master_ui's "" there. false = per-clip encode.
export function upfrontTextEncodeValue(value) {
    return value === true;
}
export function dualWidgetValue(name, value) {
    const fallback = DUAL_DEFAULTS[name];
    if (typeof fallback === "boolean") return typeof value === "boolean" ? value : fallback;
    return Number.isInteger(value) && value >= 1 && value <= 6 ? value : fallback;
}

// The single "pass 2 steps" panel row: with dual sampling it binds the split step, otherwise the refine schedule.
export function pass2RowBinding(dualActive, splitStep = 2) {
    if (dualActive) {
        const k = Number(splitStep);
        return {
            widget: "dual_split_step",
            hint: `Split at step ${k}: ${k} HyperFlow steps at pass-1 size, then ${8 - k} at pass-2 size. The refine LoRA is not used.`,
        };
    }
    return {
        widget: "pass2_steps",
        hint: "Refine schedule. 0 = same as engine steps. Tail length = round(steps x refine denoise).",
    };
}
