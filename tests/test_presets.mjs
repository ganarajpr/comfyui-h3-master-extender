import assert from "node:assert/strict";
import fs from "node:fs";
import { MASTER_PRESETS, matchPreset } from "../web/presets.js";

const inputs = JSON.parse(fs.readFileSync(new URL("./fixtures/master_object_info_inputs.json", import.meta.url), "utf8"));

assert.equal(new Set(MASTER_PRESETS.map(p => p.id)).size, MASTER_PRESETS.length, "unique ids");
assert.deepEqual(MASTER_PRESETS.map(p => p.id), ["standard", "pika_pih3_refine_2", "pika_pih3_refine_4"]);
assert.equal(MASTER_PRESETS[0].label, "Standard \u00b7 Pika + DMAD", "Standard stays first");
assert.deepEqual(MASTER_PRESETS.map(p => p.label).slice(1), ["Pika + pi-H3 refine 2-step", "Pika + pi-H3 refine 4-step"]);

// every preset value is valid for the live node
for (const p of MASTER_PRESETS) {
    for (const [name, value] of Object.entries(p.values)) {
        const spec = inputs[name];
        assert(spec, `${p.id}: unknown widget ${name}`);
        if (spec.options) assert(spec.options.includes(value), `${p.id}: ${name}=${JSON.stringify(value)} not an option`);
        else if (spec.type === "BOOLEAN") assert.equal(typeof value, "boolean", `${p.id}: ${name}`);
        else {
            assert.equal(typeof value, "number", `${p.id}: ${name}`);
            if (spec.min !== undefined) assert(value >= spec.min, `${p.id}: ${name} below min`);
            if (spec.max !== undefined) assert(value <= spec.max, `${p.id}: ${name} above max`);
        }
    }
}

// exact match -> that preset; any single change -> Custom (null)
for (const p of MASTER_PRESETS) {
    const get = (n) => p.values[n];
    assert.equal(matchPreset(get)?.id, p.id, `${p.id} matches itself`);
    for (const name of Object.keys(p.values)) {
        const v = p.values[name];
        const changed = typeof v === "boolean" ? !v : typeof v === "number" ? v + 1 : v + "x";
        const m = matchPreset((n) => (n === name ? changed : p.values[n]));
        assert.notEqual(m?.id, p.id, `${p.id}: changing ${name} must not still match`);
    }
}
assert.equal(matchPreset(() => undefined), null, "missing widgets -> Custom");

// the three presets are told apart, also after "selecting" one onto a node that held another (every listed widget is written)
const [standard, refine2, refine4] = MASTER_PRESETS;
assert.equal(standard.values.pass2_engine, "LoRA refine");
const REFINE = "pi-H3 refine (DWD, no VAE decode)";
for (const p of [refine2, refine4]) {
    assert.equal(p.values.pass2_engine, REFINE);
    for (const k of ["accel_mode", "turbo_lora", "turbo_lora_strength", "pdd_nfe", "turbo_sampler", "turbo_scheduler", "pass1_resolution", "pass2_resolution"]) {
        assert.equal(p.values[k], standard.values[k], `${p.id}: ${k} is the Standard pass-1 value`);
    }
}
assert.deepEqual([refine2.values.pih3_refine_steps, refine2.values.pih3_refine_tail], [2, 1]);
assert.deepEqual([refine4.values.pih3_refine_steps, refine4.values.pih3_refine_tail], [4, 2]);
let node = {};
for (const from of MASTER_PRESETS) {
    for (const to of MASTER_PRESETS) {
        node = { ...from.values, pass2_pass: "kept" };
        Object.assign(node, to.values);
        assert.equal(matchPreset((n) => node[n])?.id, to.id, `${from.id} -> ${to.id}`);
    }
}
// a refine workflow is not Standard, and a Standard workflow from before pass2_engine existed (repaired to LoRA refine) still is
assert.notEqual(matchPreset((n) => ({ ...standard.values, pass2_engine: REFINE })[n])?.id, "standard");
assert.equal(matchPreset((n) => ({ ...standard.values, pass2_engine: "LoRA refine" })[n])?.id, "standard");
for (const hint of MASTER_PRESETS.slice(1).map(p => p.hint)) assert(hint && !hint.includes("\n"), "one-line hint");
assert(!/\d+(\.\d+)? ?s\b/.test(refine4.hint), "no timing claim for the 4-step");

// loading never writes: run the matcher over every example workflow's widgets_values
const dir = new URL("../example_workflows/", import.meta.url);
for (const file of fs.readdirSync(dir).filter(f => f.endsWith(".json"))) {
    const node = JSON.parse(fs.readFileSync(new URL(file, dir), "utf8")).nodes.find(n => n.type === "MiniMaxH3MasterExtender");
    const names = node.inputs.filter(i => i.widget).map(i => i.name);
    const before = JSON.stringify(node.widgets_values);
    const widgets = Object.freeze(Object.fromEntries(names.map((n, i) => [n, node.widgets_values[i]])));
    matchPreset((n) => widgets[n]);
    assert.equal(JSON.stringify(node.widgets_values), before, `${file}: values unchanged on load`);
}

// the panel only imports presets.js behind try/catch and reads through matchPreset on render
const js = fs.readFileSync(new URL("../web/master_extender.js", import.meta.url), "utf8");
assert(/try \{ presetsMod = await import\("\.\/presets\.js"\)/.test(js), "guarded dynamic import");
assert(!/from "\.\/presets\.js"/.test(js), "no static import of presets.js");

console.log("presets ok");
