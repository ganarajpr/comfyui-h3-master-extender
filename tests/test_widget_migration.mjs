import assert from "node:assert/strict";
import fs from "node:fs";
import { dropCaptionModelValue, dropSingleTextEncodeValue, normalizeReasoningBudget, migrateAccelMode, dualWidgetValue, DUAL_DEFAULTS } from "../web/widget_migration.js";

const workflows = fs.readdirSync(new URL("../example_workflows/", import.meta.url)).filter(f => f.endsWith(".json"));
assert(workflows.length >= 5);

for (const file of workflows) {
    const flow = JSON.parse(fs.readFileSync(new URL(`../example_workflows/${file}`, import.meta.url), "utf8"));
    const node = flow.nodes.find(n => n.type === "MiniMaxH3MasterExtender");
    // widget order of the current node = its widget inputs, then the JS-only master_ui
    const DUAL = Object.keys(DUAL_DEFAULTS);
    const allNames = [...node.inputs.filter(i => i.widget).map(i => i.name), "master_ui"];
    const allValues = node.widgets_values;
    // dual sampling widgets sit between sparse_sink and master_ui; the older layout assertions below run without them
    const hasDual = allNames.includes("dual_sampling");
    if (hasDual) assert.deepEqual(allNames.slice(-4), [...DUAL, "master_ui"].slice(0, 3).concat("master_ui"), `${file}: dual widgets are last`);
    const names = allNames.filter(n => !DUAL.includes(n));
    const current = allValues.filter((_, i) => !DUAL.includes(allNames[i]));
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

// dual sampling widgets: old arrays leave them undefined or hold master_ui's "" -> defaults; real values are kept
for (const name of Object.keys(DUAL_DEFAULTS)) {
    for (const bad of [undefined, "", null, "x", 0, 7, 2.5]) assert.equal(dualWidgetValue(name, bad), DUAL_DEFAULTS[name], `${name} <- ${String(bad)}`);
}
assert.equal(dualWidgetValue("dual_sampling", true), true);
assert.equal(dualWidgetValue("pass1_early_stop", true), true);
assert.equal(dualWidgetValue("dual_split_step", 6), 6);
assert.deepEqual(DUAL_DEFAULTS, { dual_sampling: false, dual_split_step: 2, pass1_early_stop: false });
{
    const flow = JSON.parse(fs.readFileSync(new URL("../example_workflows/MiniMax_H3_Master_Extender_HyperFlow_DualSampling.json", import.meta.url), "utf8"));
    const node = flow.nodes.find(n => n.type === "MiniMaxH3MasterExtender");
    const names = [...node.inputs.filter(i => i.widget).map(i => i.name), "master_ui"];
    assert.equal(names.length, node.widgets_values.length);
    const get = (n) => node.widgets_values[names.indexOf(n)];
    assert.deepEqual([get("dual_sampling"), get("dual_split_step"), get("pass1_early_stop")], [true, 2, false]);
    assert.equal(get("pass2_lora"), "none");
    assert.equal(get("accel_mode"), "HyperFlow 8-step");
    assert.equal(get("pass1_resolution"), "608x352 (16:9)");
    assert.equal(get("pass2_resolution"), "1344x768 (16:9)");
}
assert.equal(dropCaptionModelValue(["rewrite_writer_model", "rewrite_caption_length"], undefined), null);

const table = [[-1, "4096"], [0, "4096"], [512, "1024"], [1024, "1024"], [1500, "1024"], [1536, "2048"], [2048, "2048"],
    [3072, "4096"], [4096, "4096"], [8192, "4096"], [16384, "4096"], [32768, "4096"], ["2048", "2048"], ["", "4096"], [null, "4096"], ["x", "4096"]];
for (const [given, want] of table) assert.equal(normalizeReasoningBudget(given), want, String(given));
const accelNames = ["pdd_file", "accel_mode", "pdd_nfe"];
const saved = ["minimax_h3_ref2va_pdd_acc_8step_comfyui.safetensors", "PDD 8-step", "8"];
assert.deepEqual(migrateAccelMode(accelNames, saved), [saved[0], "HyperFlow 8-step", "8"]);
assert.deepEqual(saved[1], "PDD 8-step", "input array is not mutated");
assert.equal(migrateAccelMode(accelNames, [saved[0], "HyperFlow 8-step", "8"]), null);
assert.equal(migrateAccelMode(accelNames, [saved[0], "Turbo LoRA", "4"]), null);
assert.equal(migrateAccelMode(["pdd_file"], ["x"]), null);
assert.equal(migrateAccelMode(accelNames, undefined), null);
for (const file of workflows) {
    const flow = JSON.parse(fs.readFileSync(new URL(`../example_workflows/${file}`, import.meta.url), "utf8"));
    const node = flow.nodes.find(n => n.type === "MiniMaxH3MasterExtender");
    const names = node.inputs.filter(i => i.widget).map(i => i.name);
    assert.equal(migrateAccelMode(names, node.widgets_values), null, `${file}: no PDD left`);
    assert(["HyperFlow 8-step", "Turbo LoRA", "HyperFlow 8-step \u00b7 dual sampling"].includes(node.widgets_values[names.indexOf("accel_mode")]), file);
}
console.log("widget migration ok");

// pass 2 steps row: dual on -> dual_split_step, off -> pass2_steps; binding never writes
{
    const { pass2RowBinding } = await import("../web/widget_migration.js");
    assert.equal(pass2RowBinding(true, 3).widget, "dual_split_step");
    assert.match(pass2RowBinding(true, 3).hint, /3 HyperFlow steps.*then 5/);
    assert.equal(pass2RowBinding(false, 3).widget, "pass2_steps");
    assert.match(pass2RowBinding(false).hint, /Refine schedule/);
    const dir = new URL("../example_workflows/", import.meta.url);
    for (const file of fs.readdirSync(dir).filter(f => f.endsWith(".json"))) {
        const node = JSON.parse(fs.readFileSync(new URL(file, dir), "utf8")).nodes.find(n => n.type === "MiniMaxH3MasterExtender");
        const before = JSON.stringify(node.widgets_values);
        const frozen = Object.freeze([...node.widgets_values]);
        pass2RowBinding(true, frozen[0]); pass2RowBinding(false, frozen[0]);
        assert.equal(JSON.stringify(node.widgets_values), before, `${file}: pass2 row binding leaves values unchanged`);
    }
}

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
