import assert from "node:assert/strict";
import fs from "node:fs";
import { MASTER_PRESETS, matchPreset } from "../web/presets.js";

const inputs = JSON.parse(fs.readFileSync(new URL("./fixtures/master_object_info_inputs.json", import.meta.url), "utf8"));

assert.equal(new Set(MASTER_PRESETS.map(p => p.id)).size, MASTER_PRESETS.length, "unique ids");
assert.equal(MASTER_PRESETS.length, 12);

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
