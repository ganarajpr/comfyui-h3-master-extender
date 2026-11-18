import assert from "node:assert/strict";
import fs from "node:fs";
import { dropCaptionModelValue, dropHyperflowValues, dropDualSamplingValues, takeHyperflowValues, takeDualSamplingValues, migrateSavedValues, tstOffForSavedValues, dualWidgetValue, pass1StepsValue, hyperflowWidgetValue, isHyperflowMode, LEGACY_HYPERFLOW_WIDGETS, dropSingleTextEncodeValue, normalizeReasoningBudget, migrateAccelMode, isLegacyAccelMode, flashStepsValue, flashFileValue, flashLoraValue, flashLoraStrengthValue, handoffStepValue, HANDOFF_STEP_OPTIONS } from "../web/widget_migration.js";

const workflows = fs.readdirSync(new URL("../example_workflows/", import.meta.url)).filter(f => f.endsWith(".json"));
assert(workflows.length >= 5);
assert(!workflows.some(f => /DualSampling/.test(f)), "the dual sampling examples stay deleted: HyperFlow 8-step is legacy, no example is built on it");

for (const file of workflows) {
    const flow = JSON.parse(fs.readFileSync(new URL(`../example_workflows/${file}`, import.meta.url), "utf8"));
    const node = flow.nodes.find(n => n.type === "MiniMaxH3MasterExtender");
    // widget order of the current node = its widget inputs, then the JS-only master_ui
    const names = [...node.inputs.filter(i => i.widget).map(i => i.name), "master_ui"];
    const current = node.widgets_values;
    for (const gone of ["hyperflow_file", "hyperflow_curve_refit", "hyperflow_strength", "hyperflow_lora_mode", "dual_sampling", "dual_split_step", "pass1_early_stop", "pass1_steps"]) {
        assert(!names.includes(gone), `${file}: ${gone}: the example workflows were saved without the legacy widgets`);
    }
    assert.deepEqual(names.slice(-5), ["e4_picture_notes", "sparse_start_percent", "sparse_dense_blocks", "sparse_sink", "master_ui"], `${file}: sparse widgets are last`);
    assert.deepEqual(current.slice(-4), [-1, "auto", "auto", ""], file);
    assert.equal(names.length, current.length, file);
    assert.equal(dropCaptionModelValue(names, current), null, `${file}: current layout is left alone`);
    assert.equal(dropSingleTextEncodeValue(names, current), null, `${file}: no single_text_encode value to drop`);

    // the same workflow as saved before single_text_encode was removed: one boolean right after audio_refine_cache
    const cache = names.indexOf("audio_refine_cache");
    for (const flag of [true, false]) {
        const withFlag = [...current];
        withFlag.splice(cache + 1, 0, flag);
        const fixed = dropSingleTextEncodeValue(names, withFlag);
        assert.deepEqual(fixed, current, `${file}: single_text_encode=${flag} dropped, later widgets re-seated`);
    }

    // the same workflow as saved before the merge: one extra value right after the writer, an INT budget
    const w = names.indexOf("rewrite_writer_model");
    const old = [...current];
    old.splice(w + 1, 0, "Some Captioner (local) · with its bf16 mmproj");
    old[names.indexOf("rewrite_reasoning_budget") + 1] = 4096;
    const migrated = dropCaptionModelValue(names, old);
    assert.deepEqual(migrated.slice(0, 42), current.slice(0, 42), file);
    assert.equal(migrated.length, current.length, file);
    assert.equal(migrated[names.indexOf("rewrite_caption_length")], current[names.indexOf("rewrite_caption_length")], file);
    assert.equal(migrated[names.indexOf("rewrite_story")], current[names.indexOf("rewrite_story")], file);
    assert.equal(migrated[names.indexOf("auto_clip_seconds")], 15, file);
    assert.equal(migrated[names.indexOf("master_ui")], "", file);
    assert.equal(normalizeReasoningBudget(migrated[names.indexOf("rewrite_reasoning_budget")]), "4096");

    // an old workflow saved before the trailing widgets existed: shorter array, master_ui's "" in a new slot
    const older = old.slice(0, names.indexOf("rewrite_story") + 1 + 1);
    const olderMigrated = dropCaptionModelValue(names, older);
    assert.equal(olderMigrated.length, older.length - 1);
    assert.equal(olderMigrated[names.indexOf("rewrite_caption_length")], current[names.indexOf("rewrite_caption_length")]);
}

assert.equal(dropCaptionModelValue(["rewrite_writer_model"], ["w"]), null);

assert.equal(dropCaptionModelValue(["rewrite_writer_model", "rewrite_caption_length"], undefined), null);

const table = [[-1, "4096"], [0, "4096"], [512, "1024"], [1024, "1024"], [1500, "1024"], [1536, "2048"], [2048, "2048"],
    [3072, "4096"], [4096, "4096"], [8192, "4096"], [16384, "4096"], [32768, "4096"], ["2048", "2048"], ["", "4096"], [null, "4096"], ["x", "4096"]];
for (const [given, want] of table) assert.equal(normalizeReasoningBudget(given), want, String(given));
const accelNames = ["pdd_file", "accel_mode", "pdd_nfe", "turbo_lora", "turbo_lora_strength", "turbo_sampler", "turbo_scheduler", "pass2_lora", "pass2_lora_strength", "pass2_lora_mode", "pass2_steps"];
const saved = (mode, p2 = "none", p2steps = 0) => ["x.safetensors", mode, "8", "none", 1.0, "res_multistep", "simple", p2, 1.0, "stack on engine LoRA", p2steps];
// PDD 8-step still maps to HyperFlow 8-step and nothing else in the array changes
{
    const input = saved("PDD 8-step");
    const out = migrateAccelMode(accelNames, input);
    assert.deepEqual(out, saved("HyperFlow 8-step"));
    assert.equal(input[1], "PDD 8-step", "input array is not mutated");
}
// HyperFlow 8-step (plain and dual sampling) is a legacy engine again: nothing is rewritten, no Pika / DMAD is filled in
for (const keep of ["HyperFlow 8-step", "HyperFlow 8-step \u00b7 dual sampling", "Turbo LoRA", "HyperFlow Flash (GGUF)"]) assert.equal(migrateAccelMode(accelNames, saved(keep)), null, keep);
assert.equal(migrateAccelMode(["pdd_file"], ["x"]), null);
assert.equal(migrateAccelMode(accelNames, undefined), null);
assert(isLegacyAccelMode("PDD 8-step") && !isLegacyAccelMode("HyperFlow 8-step") && !isLegacyAccelMode("HyperFlow Flash (GGUF)") && !isLegacyAccelMode("Turbo LoRA"));
assert(isHyperflowMode("HyperFlow 8-step") && isHyperflowMode("HyperFlow 8-step \u00b7 dual sampling") && !isHyperflowMode("HyperFlow Flash (GGUF)") && !isHyperflowMode("Turbo LoRA"));
for (const file of workflows) {
    const flow = JSON.parse(fs.readFileSync(new URL(`../example_workflows/${file}`, import.meta.url), "utf8"));
    const node = flow.nodes.find(n => n.type === "MiniMaxH3MasterExtender");
    const names = node.inputs.filter(i => i.widget).map(i => i.name);
    assert.equal(migrateAccelMode(names, node.widgets_values), null, `${file}: no PDD mode left`);
    assert(["Turbo LoRA", "HyperFlow Flash (GGUF)"].includes(node.widgets_values[names.indexOf("accel_mode")]), file);
}

// A saved UI workflow carries the legacy HyperFlow widgets as extra values: removed from the array (later widgets re-seat) and returned by name.
{
    const names = ["rewrite_previous_clips", "audio_refine_steps", "audio_refine_denoise", "sparse_sink", "pass2_chunk_when", "pass2_sla_sparsity", "master_ui"];
    const now = ["raw asks", 0, 0.5, "auto", "above 720p", 0, ""];
    assert.equal(dropHyperflowValues(names, now), null, "current layout untouched (hyperflow block)");
    assert.equal(dropDualSamplingValues(names, now), null, "current layout untouched (dual block)");
    const hf = [...now.slice(0, 1), "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors", true, 1.0, "bypass", ...now.slice(1)];
    assert.deepEqual(dropHyperflowValues(names, hf), now);
    for (const mode of ["bypass", "merge"]) assert.deepEqual(dropHyperflowValues(names, [now[0], "", false, 0.5, mode, ...now.slice(1)]), now);
    // dual block: dual_sampling, dual_split_step, pass1_early_stop (+ pass1_steps in later saves)
    const dual3 = [...now.slice(0, 4), true, 2, false, ...now.slice(4)];
    const dual4 = [...now.slice(0, 4), false, 3, true, 5, ...now.slice(4)];
    assert.deepEqual(dropDualSamplingValues(names, dual3), now);
    assert.deepEqual(dropDualSamplingValues(names, dual4), now);
    // a save that ended right after the dual widgets: master_ui's "" follows them
    assert.deepEqual(dropDualSamplingValues(names, [...now.slice(0, 4), true, 2, false, ""]), [...now.slice(0, 4), ""]);
    // both blocks at once, in the order the panel applies them
    const both = dropDualSamplingValues(names, dropHyperflowValues(names, [now[0], "f.safetensors", true, 1.0, "bypass", ...now.slice(1, 4), true, 2, false, 0, ...now.slice(4)]));
    assert.deepEqual(both, now);
    // pre-HyperFlow / pre-dual saves: nothing to drop
    assert.equal(dropHyperflowValues(names, [now[0], 4, 0.5, "auto", "", 0, ""]), null);
    assert.equal(dropDualSamplingValues(names, [...now.slice(0, 4), "", undefined]), null);
    assert.equal(dropHyperflowValues(names, undefined), null);
    assert.equal(dropDualSamplingValues(["x"], ["y"]), null);
    // the removed values come back by widget name
    assert.deepEqual(takeHyperflowValues(names, hf).legacy, { hyperflow_file: "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors", hyperflow_curve_refit: true, hyperflow_strength: 1.0, hyperflow_lora_mode: "bypass" });
    assert.deepEqual(takeDualSamplingValues(names, dual3).legacy, { dual_sampling: true, dual_split_step: 2, pass1_early_stop: false, pass1_steps: 0 }, "a 3-value dual block means pass1_steps 0");
    assert.deepEqual(takeDualSamplingValues(names, dual4).legacy, { dual_sampling: false, dual_split_step: 3, pass1_early_stop: true, pass1_steps: 5 });
    assert.equal(takeHyperflowValues(names, now), null);
    assert.equal(takeDualSamplingValues(names, now), null);
}

// HyperFlow Flash widgets (appended last): master_ui's "" in the file slot and any non-4/6/8 steps fall back to the defaults
assert.equal(flashStepsValue(""), 6);
assert.equal(flashStepsValue(5), 6);
assert.deepEqual([4, 6, 8].map(flashStepsValue), [4, 6, 8]);
assert.equal(flashLoraValue("", ["none", "a.safetensors"]), "none");
assert.equal(flashLoraValue("gone.safetensors", ["none", "a.safetensors"]), "none");
assert.equal(flashLoraValue("a.safetensors", ["none", "a.safetensors"]), "a.safetensors");
assert.equal(flashLoraStrengthValue(""), 1.0);
assert.equal(flashLoraStrengthValue(0.5), 0.5);
const flashFiles = ["a_ref2va_hyperflow_flash.gguf", "b_ref2va_hyperflow_flash.gguf"];
assert.equal(flashFileValue("", flashFiles), flashFiles[0]);
assert.equal(flashFileValue("gone.gguf", flashFiles), flashFiles[0]);
assert.equal(flashFileValue(flashFiles[1], flashFiles), flashFiles[1]);
const flashMixed = ["minimax_h3_fl2va_hyperflow_flash_14b-Q8_0.gguf", "minimax_h3_ref2va_hyperflow_flash_14b-Q8_0.gguf", "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot.safetensors"];
assert.equal(flashFileValue("", flashMixed), flashMixed[2]);
assert.equal(flashFileValue("", flashMixed.slice(0, 2)), flashMixed[1]);
const flashWithMerge = [...flashMixed, "minimax_h3_ref2va_hyperflow_flash_14b-int8_convrot_pikamerge.safetensors"];
assert.equal(flashFileValue("", flashWithMerge), flashWithMerge[3]);
assert.equal(flashFileValue(flashMixed[2], flashWithMerge), flashMixed[2]);
// handoff_step is a combo: master_ui's "" and any non-label value fall back to the default
const HS = ["1 \u00b7 Pika to 0.98", "2 \u00b7 Pika to 0.96", "3 \u00b7 Pika to 0.92", "4 \u00b7 Pika to 0.86", "5 \u00b7 Pika to 0.71"];
assert.deepEqual(HANDOFF_STEP_OPTIONS, HS);
assert.equal(handoffStepValue(""), HS[2]);
assert.equal(handoffStepValue(3), HS[2]);
assert.equal(handoffStepValue("nope"), HS[2]);
assert.deepEqual(HS.map(handoffStepValue), HS);
console.log("widget migration ok");

// run_mode gained prompts_only (last option, default unchanged) and clips_in was appended as the last input: a socket, so no widget value moves
{
    const inputs = JSON.parse(fs.readFileSync(new URL("./fixtures/master_object_info_inputs.json", import.meta.url), "utf8"));
    assert.deepEqual(inputs.run_mode.options, ["clip_by_clip", "full_batch", "prompts_only"]);
    assert.equal(Object.keys(inputs).at(-1), "clips_in");
    for (const file of workflows) {
        const node = JSON.parse(fs.readFileSync(new URL(`../example_workflows/${file}`, import.meta.url), "utf8")).nodes.find(n => n.type === "MiniMaxH3MasterExtender");
        const names = node.inputs.filter(i => i.widget).map(i => i.name);
        assert(!names.includes("clips_in"), `${file}: clips_in is a socket, not a widget`);
        assert.equal(names.length + 1, node.widgets_values.length, `${file}: widget values still line up (names + master_ui)`);
        assert(["clip_by_clip", "full_batch"].includes(node.widgets_values[names.indexOf("run_mode")]), `${file}: saved run_mode still valid`);
    }
}

{
    const { pass2SchedulerValue, pass2AudioValue } = await import("../web/widget_migration.js");
    assert.equal(pass2SchedulerValue(""), "same as pass 1");
    assert.equal(pass2SchedulerValue(undefined), "same as pass 1");
    assert.equal(pass2SchedulerValue("beta"), "beta");
    assert.equal(pass2AudioValue(undefined), "pass 2 (refined)");
    assert.equal(pass2AudioValue("pass 1"), "pass 1");
    // PTD-branch workflows: ptd_adapter / pass2_engine sit in these two slots
    const scheds = ["same as pass 1", "simple", "beta"];
    assert.equal(pass2SchedulerValue("none", scheds), "same as pass 1");
    assert.equal(pass2SchedulerValue("ptd_h3_ref2va_refasymopd_8step_int8.safetensors", scheds), "same as pass 1");
    assert.equal(pass2SchedulerValue("beta", scheds), "beta");
    assert.equal(pass2AudioValue("LoRA refine"), "pass 2 (refined)");
    assert.equal(pass2AudioValue("PTD last hop"), "pass 2 (refined)");
}

// pass1_cfg / cfg_steps / negative_prompt / cfg_handoff_steps are appended after handoff_step: master_ui's "" and bad values fall back
import { pass1CfgValue, cfgStepsValue, cfgHandoffStepsValue, negativePromptValue } from "../web/widget_migration.js";
assert.equal(pass1CfgValue(""), 4.5);
assert.equal(pass1CfgValue(0.5), 4.5);
assert.equal(pass1CfgValue(1), 1);
assert.equal(pass1CfgValue(6.5), 6.5);
assert.equal(cfgStepsValue(""), 24);
assert.equal(cfgStepsValue(1), 24);
assert.equal(cfgStepsValue(30), 30);
assert.equal(cfgHandoffStepsValue(undefined), 8);
assert.equal(cfgHandoffStepsValue(5), 5);
assert.equal(negativePromptValue(undefined), "");
assert.equal(negativePromptValue("blurry"), "blurry");
console.log("cfg widget migration ok");

// ---------------------------------------------------------------------------------------------------------------------
// Real layouts. NODE_WIDGETS is the node's widget order now: the legacy HyperFlow widgets are appended after tst_tau (so every
// workflow saved since 2026-11-17 keeps its order) and master_ui, the DOM widget, is last. It is checked against master_node.py.
// ---------------------------------------------------------------------------------------------------------------------
const NODE_WIDGETS = ["pdd_file", "upscaler_model", "run_mode", "pass1_resolution", "pass2_resolution", "pass2_denoise", "pdd_nfe", "context_length", "audio_context_length", "identity_continuity", "smart_offload", "clips_json", "refs_json", "attention_backend", "sla_enabled", "sla_sparsity", "pass2_chunk_frames", "pass2_chunk_overlap", "async_decode", "sparse_method", "sparse_tau", "accel_mode", "turbo_lora", "turbo_lora_strength", "turbo_sampler", "turbo_scheduler", "pass2_lora", "pass2_lora_strength", "pass2_lora_mode", "pass2_steps", "semantic_bridge", "semantic_bridge_alpha", "semantic_bridge_match", "rewrite_mode", "rewrite_task", "rewrite_writer_model", "rewrite_caption_length", "rewrite_greedy", "rewrite_temperature", "rewrite_max_new_tokens", "rewrite_seed", "rewrite_thinking", "rewrite_reasoning_budget", "rewrite_reasoning_budget_message", "rewrite_parallel", "rewrite_system_prompt", "rewrite_previous_clips", "audio_refine_steps", "audio_refine_denoise", "audio_refine_cache", "final_decode", "rewrite_story", "auto_clips", "planner_refs", "auto_clip_seconds", "story_engine", "e4_language", "e4_score", "e4_decision_budget", "e4_picture_notes", "sparse_start_percent", "sparse_dense_blocks", "sparse_sink", "pass2_chunk_when", "pass2_sla_sparsity", "pass2_scheduler", "pass2_audio", "hyperflow_flash_file", "hyperflow_flash_steps", "hyperflow_flash_lora", "hyperflow_flash_lora_strength", "handoff_step", "pass1_cfg", "cfg_steps", "negative_prompt", "cfg_handoff_steps", "tst_enabled", "tst_tau", "hyperflow_file", "hyperflow_curve_refit", "hyperflow_strength", "hyperflow_lora_mode", "dual_sampling", "dual_split_step", "pass1_early_stop", "pass1_steps", "master_ui"];
{
    const src = fs.readFileSync(new URL("../master_node.py", import.meta.url), "utf8");
    const at = NODE_WIDGETS.filter(n => n !== "master_ui").map(n => src.indexOf(`"${n}": (`, src.indexOf("def INPUT_TYPES")));
    assert(at.every(i => i > 0), "every widget name is an input of master_node.py");
    assert(at.every((v, i) => i === 0 || v > at[i - 1]), "NODE_WIDGETS follows master_node.py's input order");
    assert.deepEqual(NODE_WIDGETS.slice(NODE_WIDGETS.indexOf("tst_tau") + 1), [...LEGACY_HYPERFLOW_WIDGETS, "master_ui"], "legacy widgets are appended right after tst_tau");
}
const NEW_SINCE_1117 = ["hyperflow_flash_file", "hyperflow_flash_steps", "hyperflow_flash_lora", "hyperflow_flash_lora_strength", "handoff_step", "pass1_cfg", "cfg_steps", "negative_prompt", "cfg_handoff_steps", "tst_enabled", "tst_tau"];
const DMAD = "h3\\minimax_h3_DMAD_4step_full_lora_avg_rank_39_bf16.safetensors";
const PRUNED = "custom_node_hyperflow_8step_v1.0_comfyui_pruned.safetensors";
const SAVED_VALUE = {
    accel_mode: "HyperFlow 8-step", pdd_nfe: "8", turbo_lora: "none", turbo_lora_strength: 1, turbo_sampler: "res_multistep", turbo_scheduler: "simple",
    pass2_lora: DMAD, pass2_lora_strength: 1, pass2_lora_mode: "replace engine LoRA", pass2_steps: 4,
    rewrite_caption_length: "standard", rewrite_previous_clips: "raw asks", audio_refine_steps: 0, audio_refine_denoise: 0.5, audio_refine_cache: true,
    final_decode: "standard", sparse_start_percent: -1, sparse_dense_blocks: "auto", sparse_sink: "auto", pass2_chunk_when: "above 720p", pass2_sla_sparsity: 0,
    pass2_scheduler: "none", pass2_audio: "LoRA refine", master_ui: "",
};
// the layout of a workflow saved 2026-11-17: the current widgets minus the eleven added since, with the four HyperFlow values
// after rewrite_previous_clips and the four dual sampling values after sparse_sink
function layout1117(overrides = {}, dual = [false, 2, false, 0], hf = [PRUNED, true, 1, "bypass"]) {
    const base = NODE_WIDGETS.filter(n => !NEW_SINCE_1117.includes(n) && !LEGACY_HYPERFLOW_WIDGETS.includes(n));
    const names = [], values = [];
    const put = (name, value) => { names.push(name); values.push(value); };
    for (const name of base) {
        put(name, name in overrides ? overrides[name] : name in SAVED_VALUE ? SAVED_VALUE[name] : `v:${name}`);
        if (name === "rewrite_previous_clips") hf.forEach((v, i) => put(LEGACY_HYPERFLOW_WIDGETS[i], v));
        if (name === "sparse_sink") dual.forEach((v, i) => put(LEGACY_HYPERFLOW_WIDGETS[4 + i], v));
    }
    return { names, values };
}
// What the panel does on load: LiteGraph seats the array by index, then onConfigure re-seats it and repairs the widgets.
function loadLikeTheNode(saved) {
    const names = NODE_WIDGETS;
    const widgets = Object.fromEntries(names.map(n => [n, undefined]));
    widgets.tst_enabled = true; widgets.tst_tau = 0.2;
    saved.forEach((v, i) => { if (i < names.length) widgets[names[i]] = v; });
    const info = { widgets_values: saved };
    const result = migrateSavedValues(names, saved);
    if (result) {
        info.widgets_values = result.values;
        result.values.forEach((v, i) => { widgets[names[i]] = v; });
        Object.assign(widgets, result.legacy);
    }
    for (const n of ["dual_sampling", "dual_split_step", "pass1_early_stop", "pass1_steps"]) widgets[n] = dualWidgetValue(n, widgets[n]);
    for (const n of ["hyperflow_file", "hyperflow_curve_refit", "hyperflow_strength", "hyperflow_lora_mode"]) widgets[n] = hyperflowWidgetValue(n, widgets[n], PRUNED);
    if (typeof widgets.tst_enabled !== "boolean") widgets.tst_enabled = false;
    if (tstOffForSavedValues(names, info.widgets_values)) widgets.tst_enabled = false;
    return { widgets, info };
}

// A 76-value array saved 2026-11-17 on HyperFlow 8-step (PTD-branch values in the pass2_scheduler / pass2_audio slots, master_ui last)
{
    const { names, values } = layout1117();
    assert.equal(values.length, 76);
    assert.equal(names.at(-1), "master_ui");
    const { widgets, info } = loadLikeTheNode(values);
    assert.equal(widgets.accel_mode, "HyperFlow 8-step", "the engine is NOT rewritten to Turbo LoRA");
    assert.equal(widgets.pdd_nfe, "8");
    assert.equal(widgets.turbo_lora, "none", "turbo settings untouched");
    assert.equal(widgets.turbo_sampler, "res_multistep");
    assert.equal(widgets.turbo_scheduler, "simple");
    assert.equal(widgets.pass2_lora, DMAD);
    assert.equal(widgets.pass2_lora_mode, "replace engine LoRA");
    assert.equal(widgets.pass2_steps, 4);
    for (const [name, want] of Object.entries({ hyperflow_file: PRUNED, hyperflow_curve_refit: true, hyperflow_strength: 1, hyperflow_lora_mode: "bypass", dual_sampling: false, dual_split_step: 2, pass1_early_stop: false, pass1_steps: 0 })) {
        assert.equal(widgets[name], want, name);
    }
    assert.equal(widgets.rewrite_previous_clips, "raw asks");
    assert.equal(widgets.audio_refine_steps, 0, "later widgets re-seated");
    assert.equal(widgets.sparse_sink, "auto");
    assert.equal(widgets.pass2_chunk_when, "above 720p");
    assert.equal(widgets.pass2_sla_sparsity, 0);
    assert.equal(widgets.tst_enabled, false, "a workflow from before TST loads with TST off");
    assert.equal(info.widgets_values.length, 68, "info keeps the shortened array: the legacy values are not written into it");
    assert(info.widgets_values.length <= NODE_WIDGETS.indexOf("tst_tau"));
    assert.deepEqual(info.widgets_values.slice(-1), [""], "master_ui's value is the last of the shortened array");
}
// the same but with non-default saved HyperFlow values and a dual sampling block with pass1_steps, on the plain HyperFlow mode
{
    const { values } = layout1117({}, [true, 5, true, 3], ["custom_node_hyperflow_8step_v1.0_comfyui.safetensors", false, 0.8, "merge"]);
    assert.equal(values.length, 76);
    const { widgets } = loadLikeTheNode(values);
    assert.equal(widgets.hyperflow_file, "custom_node_hyperflow_8step_v1.0_comfyui.safetensors");
    assert.equal(widgets.hyperflow_curve_refit, false);
    assert.equal(widgets.hyperflow_strength, 0.8);
    assert.equal(widgets.hyperflow_lora_mode, "merge");
    assert.deepEqual([widgets.dual_sampling, widgets.dual_split_step, widgets.pass1_early_stop, widgets.pass1_steps], [true, 5, true, 3]);
    assert.equal(widgets.accel_mode, "HyperFlow 8-step");
}
// A 70-value array that ends in the dual block (true, 7, true), no master_ui value, on the dual sampling mode
{
    const { values } = layout1117({ accel_mode: "HyperFlow 8-step \u00b7 dual sampling" });
    const cut = values.slice(0, values.indexOf("above 720p"));
    assert.deepEqual(cut.slice(-4), [false, 2, false, 0]);
    const sevens = [...cut.slice(0, -4), true, 7, true];
    assert.equal(sevens.length, 70);
    const { widgets, info } = loadLikeTheNode(sevens);
    assert.equal(widgets.accel_mode, "HyperFlow 8-step \u00b7 dual sampling");
    assert.deepEqual([widgets.dual_sampling, widgets.dual_split_step, widgets.pass1_early_stop, widgets.pass1_steps], [true, 7, true, 0], "3-value dual block: pass1_steps 0, split 7 kept");
    assert.equal(widgets.hyperflow_file, PRUNED);
    assert.equal(widgets.turbo_lora, "none");
    assert.equal(widgets.tst_enabled, false);
    assert.equal(info.widgets_values.length, 63);
}
// A workflow saved 2026-11-17 / 18 on Turbo LoRA (the layout before these widgets were appended): master_ui's "" lands in hyperflow_file, the rest is missing
{
    const base = NODE_WIDGETS.filter(n => !LEGACY_HYPERFLOW_WIDGETS.includes(n) && n !== "master_ui");
    const values = base.map(n => n in SAVED_VALUE ? SAVED_VALUE[n] : `v:${n}`);
    values[base.indexOf("accel_mode")] = "Turbo LoRA";
    values[base.indexOf("tst_enabled")] = true;
    values[base.indexOf("tst_tau")] = 0.3;
    values[base.indexOf("pass2_scheduler")] = "same as pass 1";
    values[base.indexOf("pass2_audio")] = "pass 2 (refined)";
    values.push("");
    assert.equal(migrateSavedValues(NODE_WIDGETS, values), null, "nothing positional to move");
    const { widgets } = loadLikeTheNode(values);
    assert.equal(widgets.accel_mode, "Turbo LoRA");
    assert.equal(widgets.tst_enabled, true, "TST as saved");
    assert.equal(widgets.tst_tau, 0.3);
    assert.equal(widgets.hyperflow_file, PRUNED, "master_ui's empty value is repaired to the default build");
    assert.deepEqual([widgets.hyperflow_curve_refit, widgets.hyperflow_strength, widgets.hyperflow_lora_mode], [true, 1, "bypass"]);
    assert.deepEqual([widgets.dual_sampling, widgets.dual_split_step, widgets.pass1_early_stop, widgets.pass1_steps], [false, 2, false, 0]);
}
// repair functions: wrong types fall back, a non-empty hyperflow_file is never swapped
assert.equal(dualWidgetValue("dual_sampling", ""), false);
assert.equal(dualWidgetValue("dual_sampling", true), true);
assert.equal(dualWidgetValue("dual_split_step", ""), 2);
assert.equal(dualWidgetValue("dual_split_step", 7), 7);
assert.equal(dualWidgetValue("dual_split_step", 8), 2);
assert.equal(dualWidgetValue("pass1_early_stop", undefined), false);
assert.equal(pass1StepsValue(""), 0);
assert.equal(pass1StepsValue(8), 8);
assert.equal(pass1StepsValue(9), 0);
assert.equal(hyperflowWidgetValue("hyperflow_file", "", "d.safetensors"), "d.safetensors");
assert.equal(hyperflowWidgetValue("hyperflow_file", "gone.safetensors", "d.safetensors"), "gone.safetensors");
assert.equal(hyperflowWidgetValue("hyperflow_curve_refit", "", "d"), true);
assert.equal(hyperflowWidgetValue("hyperflow_curve_refit", false, "d"), false);
assert.equal(hyperflowWidgetValue("hyperflow_strength", "", "d"), 1);
assert.equal(hyperflowWidgetValue("hyperflow_strength", 0.5, "d"), 0.5);
assert.equal(hyperflowWidgetValue("hyperflow_lora_mode", "", "d"), "bypass");
assert.equal(hyperflowWidgetValue("hyperflow_lora_mode", "merge", "d"), "merge");
console.log("legacy HyperFlow layout migration ok");
