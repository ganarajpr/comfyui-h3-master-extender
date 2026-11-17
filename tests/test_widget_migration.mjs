import assert from "node:assert/strict";
import fs from "node:fs";
import { dropCaptionModelValue, dropHyperflowValues, dropDualSamplingValues, dropSingleTextEncodeValue, normalizeReasoningBudget, migrateAccelMode, isLegacyAccelMode, PIKA_LORA, DMAD_LORA, flashStepsValue, flashFileValue, flashLoraValue, flashLoraStrengthValue, handoffStepValue } from "../web/widget_migration.js";

const workflows = fs.readdirSync(new URL("../example_workflows/", import.meta.url)).filter(f => f.endsWith(".json"));
assert(workflows.length >= 5);
assert(!workflows.some(f => /DualSampling/.test(f)), "the dual sampling examples were removed with HyperFlow 8-step");

for (const file of workflows) {
    const flow = JSON.parse(fs.readFileSync(new URL(`../example_workflows/${file}`, import.meta.url), "utf8"));
    const node = flow.nodes.find(n => n.type === "MiniMaxH3MasterExtender");
    // widget order of the current node = its widget inputs, then the JS-only master_ui
    const names = [...node.inputs.filter(i => i.widget).map(i => i.name), "master_ui"];
    const current = node.widgets_values;
    for (const gone of ["hyperflow_file", "hyperflow_curve_refit", "hyperflow_strength", "hyperflow_lora_mode", "dual_sampling", "dual_split_step", "pass1_early_stop", "pass1_steps"]) {
        assert(!names.includes(gone), `${file}: ${gone} is gone`);
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
const RECIPE = [PIKA_LORA, 1.0, "euler", "simple", DMAD_LORA, 1.0, "replace engine LoRA", 4];
// the settings of a saved HyperFlow 8-step node: no turbo LoRA, the ignored 8 / res_multistep defaults, no pass-2 LoRA, 0 steps
const hyperflowSaved = (mode, p2 = "none", p2steps = 0) => ["x.safetensors", mode, "8", "none", 1.0, "res_multistep", "simple", p2, 1.0, "stack on engine LoRA", p2steps];
for (const mode of ["PDD 8-step", "HyperFlow 8-step", "HyperFlow 8-step \u00b7 dual sampling"]) {
    const saved = hyperflowSaved(mode);
    assert.deepEqual(migrateAccelMode(accelNames, saved), ["x.safetensors", "Turbo LoRA", "6", ...RECIPE], mode);
    assert.equal(saved[1], mode, "input array is not mutated");
}
// only what the old workflow left unset is filled: its own pass-2 LoRA and step count stay
assert.deepEqual(migrateAccelMode(accelNames, hyperflowSaved("HyperFlow 8-step", "h3\\other.safetensors", 3)).slice(7), ["h3\\other.safetensors", 1.0, "stack on engine LoRA", 3]);
// a turbo LoRA the workflow chose keeps its steps / sampler / scheduler
assert.deepEqual(migrateAccelMode(accelNames, ["x", "HyperFlow 8-step", "8", "h3\\mine.safetensors", 0.8, "dpmpp_2m", "beta", "none", 1.0, "stack on engine LoRA", 0]).slice(1, 7), ["Turbo LoRA", "8", "h3\\mine.safetensors", 0.8, "dpmpp_2m", "beta"]);
for (const keep of ["Turbo LoRA", "HyperFlow Flash (GGUF)"]) assert.equal(migrateAccelMode(accelNames, hyperflowSaved(keep)), null, keep);
assert.equal(migrateAccelMode(["pdd_file"], ["x"]), null);
assert.equal(migrateAccelMode(accelNames, undefined), null);
assert(isLegacyAccelMode("HyperFlow 8-step") && isLegacyAccelMode("PDD 8-step") && !isLegacyAccelMode("HyperFlow Flash (GGUF)") && !isLegacyAccelMode("Turbo LoRA"));
for (const file of workflows) {
    const flow = JSON.parse(fs.readFileSync(new URL(`../example_workflows/${file}`, import.meta.url), "utf8"));
    const node = flow.nodes.find(n => n.type === "MiniMaxH3MasterExtender");
    const names = node.inputs.filter(i => i.widget).map(i => i.name);
    assert.equal(migrateAccelMode(names, node.widgets_values), null, `${file}: no legacy mode left`);
    assert(["Turbo LoRA", "HyperFlow Flash (GGUF)"].includes(node.widgets_values[names.indexOf("accel_mode")]), file);
}

// A saved UI workflow carries the removed widgets as extra values: dropped, later widgets re-seated.
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
// handoff_step (appended after the flash widgets): master_ui's "" and anything outside 1..7 fall back to 3
assert.equal(handoffStepValue(""), 3);
assert.equal(handoffStepValue(0), 3);
assert.equal(handoffStepValue(8), 3);
assert.equal(handoffStepValue(2.5), 3);
assert.deepEqual([1, 3, 7].map(handoffStepValue), [1, 3, 7]);
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
