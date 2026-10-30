import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createProjectControls } from "./master_projects.js";
import { dropCaptionModelValue, dropSingleTextEncodeValue, normalizeReasoningBudget, migrateAccelMode, dualWidgetValue, pass1StepsValue, pass2ChunkWhenValue, pass2SlaSparsityValue, pass2RowBinding } from "./widget_migration.js";

// Optional: if presets.js fails to load the panel simply has no Preset row.
let presetsMod = null;
try { presetsMod = await import("./presets.js"); } catch (e) { console.warn("[H3 Master] presets.js not loaded; no Preset dropdown", e); }

const TARGET_NODE = "MiniMaxH3MasterExtender";
const EVENT_PROGRESS = "master_extender_progress";
const EVENT_CLIPS = "master_extender_clips";
const EVENT_REWRITE = "master_extender_rewrite";
const EVENT_PLAN = "master_extender_plan";

function alignH3Frames(sec) {
    let frames = Math.max(5, Math.round(Number(sec) * 24.0));
    while (frames % 17 !== 5) {
        frames++;
    }
    return frames;
}

// Clip duration slider: whole seconds, 5-15 s by default; the per-clip
// "go beyond" checkbox unlocks up to 30 s (longer clips need far more VRAM).
const DUR_MIN = 5;
const DUR_DEFAULT = 15;   // new clips start at the full 15 s
const DUR_SOFT_MAX = 15;
const DUR_HARD_MAX = 30;

const PROMPT_PREVIEW_LINES = 4;      // compact preview inside the clip card
const PANEL_WIDTH = 640;               // screen px, independent of canvas zoom
const PANEL_MIN_HEIGHT = 460;
const PANEL_GAP = 18;

// Known H3 prompt sections get a stronger colour than ad-hoc headings.
const H3_SECTIONS = new Set([
    "subject_definitions", "summary", "retention_analysis", "detailed_description",
    "overall_soundscape", "non_diegetic_music", "dialogue", "sound_effects",
    "camera", "style", "negative", "audio", "music", "voice", "shots",
]);

async function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(text);
        return;
    }
    const area = document.createElement("textarea");
    area.value = text;
    area.style.cssText = "position: fixed; left: -9999px;";
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
}

function escapeHtml(text) {
    return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// Inline tokens: <Picture 1> / <Subject 2> / <Audio 1> references, [Shot 3] and
// [reference generation] tags, 00:04.500 timestamps, "quoted dialogue".
function highlightInline(text) {
    const re = /(<(?:Picture|Subject|Audio|Image|Ref)\s*\d*>)|(\[[^\]\n]{1,40}\])|(\b\d{1,2}:\d{2}(?:\.\d{1,3})?\b|\b\d+(?:\.\d+)?s\b)|("[^"\n]{1,200}")|(\*\*[^*\n]+\*\*)/g;
    let out = "";
    let last = 0;
    for (const m of text.matchAll(re)) {
        out += escapeHtml(text.slice(last, m.index));
        const tok = escapeHtml(m[0]);
        if (m[1]) out += `<span class="h3-ref">${tok}</span>`;
        else if (m[2]) out += `<span class="h3-tag">${tok}</span>`;
        else if (m[3]) out += `<span class="h3-time">${tok}</span>`;
        else if (m[4]) out += `<span class="h3-quote">${tok}</span>`;
        else out += `<span class="h3-bold">${tok}</span>`;
        last = m.index + m[0].length;
    }
    return out + escapeHtml(text.slice(last));
}

// Line-level structure: # title, ## section, "- key: value" definitions.
function highlightH3(text) {
    const lines = (text || "").split("\n");
    const html = lines.map((line) => {
        let m;
        if ((m = /^(\s*)(#{1,6})(\s+)(.*)$/.exec(line))) {
            const name = m[4].trim().toLowerCase();
            const cls = m[2].length >= 2 && H3_SECTIONS.has(name) ? "h3-section" : "h3-heading";
            return `${m[1]}<span class="${cls}"><span class="h3-hash">${m[2]}</span>${m[3]}${highlightInline(m[4])}</span>`;
        }
        if ((m = /^(\s*[-*]\s+)([A-Za-z0-9_][\w .\-]{0,40}?)(:)(.*)$/.exec(line))) {
            return `${escapeHtml(m[1])}<span class="h3-key">${escapeHtml(m[2])}</span><span class="h3-colon">${m[3]}</span>${highlightInline(m[4])}`;
        }
        if ((m = /^(\s*[-*]\s+)(.*)$/.exec(line))) {
            return `<span class="h3-bullet">${escapeHtml(m[1])}</span>${highlightInline(m[2])}`;
        }
        return highlightInline(line);
    });
    // Trailing newline needs a visible line so the overlay height matches the textarea.
    return html.join("\n") + "\n";
}

const H3_HIGHLIGHT_CSS = `
.minimax-prompt-panel .h3-heading { color: #c4b5fd; font-weight: 700; }
.minimax-prompt-panel .h3-section { color: #9d8cff; font-weight: 700; }
.minimax-prompt-panel .h3-hash { color: #5b5b7a; font-weight: 400; }
.minimax-prompt-panel .h3-key { color: #7dd3fc; font-weight: 600; }
.minimax-prompt-panel .h3-colon { color: #5b5b7a; }
.minimax-prompt-panel .h3-bullet { color: #5b5b7a; }
.minimax-prompt-panel .h3-ref { color: #fbbf24; font-weight: 600; }
.minimax-prompt-panel .h3-tag { color: #4ade80; }
.minimax-prompt-panel .h3-time { color: #f472b6; }
.minimax-prompt-panel .h3-quote { color: #fcd9a8; font-style: italic; }
.minimax-prompt-panel .h3-bold { color: #ffffff; font-weight: 700; }
.minimax-prompt-panel textarea::selection { background: rgba(157, 140, 255, 0.35); }
.minimax-prompt-preview .h3-heading, .minimax-prompt-preview .h3-section { color: #b8a9ff; font-weight: 600; }
.minimax-prompt-preview .h3-hash, .minimax-prompt-preview .h3-colon, .minimax-prompt-preview .h3-bullet { color: #5b5b7a; }
.minimax-prompt-preview .h3-key { color: #7dd3fc; }
.minimax-prompt-preview .h3-ref { color: #fbbf24; }
.minimax-prompt-preview .h3-tag { color: #4ade80; }
.minimax-prompt-preview .h3-time { color: #f472b6; }
`;

function ensureHighlightStyles() {
    if (document.getElementById("minimax-h3-highlight-css")) return;
    const style = document.createElement("style");
    style.id = "minimax-h3-highlight-css";
    style.textContent = H3_HIGHLIGHT_CSS;
    document.head.appendChild(style);
}

// Card preview: the "## summary" section when the prompt has one, else the
// prompt from the top. Blank lines are dropped so the four visible lines count.
function previewText(prompt) {
    const lines = (prompt || "").split(/\r?\n/);
    const start = lines.findIndex((l) => /^\s*#{1,6}\s+summary\s*$/i.test(l));
    if (start >= 0) {
        const body = [];
        for (let i = start + 1; i < lines.length; i++) {
            if (/^\s*#{1,6}\s+\S/.test(lines[i])) break;
            if (lines[i].trim()) body.push(lines[i]);
        }
        if (body.length) return { text: body.join("\n"), fromSummary: true };
    }
    return { text: lines.filter((l) => l.trim()).join("\n"), fromSummary: false };
}

function describePrompt(text) {
    const trimmed = (text || "").trim();
    if (!trimmed) return "empty";
    const words = trimmed.split(/\s+/).length;
    const lines = trimmed.split(/\r?\n/).length;
    return `${words} words · ${lines} lines`;
}

// Screen-space rectangle of a node, from LiteGraph's canvas transform.
function nodeScreenRect(node) {
    const canvas = app.canvas;
    const rect = canvas.canvas.getBoundingClientRect();
    const ds = canvas.ds;
    const titleH = (window.LiteGraph && window.LiteGraph.NODE_TITLE_HEIGHT) || 30;
    const x = rect.left + (node.pos[0] + ds.offset[0]) * ds.scale;
    const y = rect.top + (node.pos[1] - titleH + ds.offset[1]) * ds.scale;
    return { x, y, w: node.size[0] * ds.scale, h: (node.size[1] + titleH) * ds.scale, canvasRect: rect };
}

// Place the panel beside the node: right of it when there is room, else left,
// else clamped inside the canvas. Height follows the node but stays readable.
function placePanelNextToNode(panel, node) {
    const r = nodeScreenRect(node);
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const height = Math.max(PANEL_MIN_HEIGHT, Math.min(r.h, vh - 40));
    let x = r.x + r.w + PANEL_GAP;
    if (x + PANEL_WIDTH > vw - 10) x = r.x - PANEL_GAP - PANEL_WIDTH;
    if (x < 10) x = Math.max(10, Math.min(vw - PANEL_WIDTH - 10, r.x + r.w + PANEL_GAP));
    let y = r.y;
    if (y + height > vh - 10) y = vh - 10 - height;
    if (y < 10) y = 10;
    panel.style.left = `${Math.round(x)}px`;
    panel.style.top = `${Math.round(y)}px`;
    panel.style.width = `${PANEL_WIDTH}px`;
    panel.style.height = `${Math.round(height)}px`;
}

function clampDuration(sec, beyond) {
    const max = beyond ? DUR_HARD_MAX : DUR_SOFT_MAX;
    const n = Math.round(Number(sec));
    if (!Number.isFinite(n)) return DUR_MIN;
    return Math.min(max, Math.max(DUR_MIN, n));
}

// Conditional widget visibility on the Master node. A hidden widget keeps its
// value and its slot in widgets_values; it just draws with zero height.
const TURBO_ONLY_WIDGETS = ["turbo_lora", "turbo_lora_strength", "turbo_sampler", "turbo_scheduler"];
// pdd_file is retired: it keeps its slot in widgets_values (saved workflows) but is never shown.
const ALWAYS_HIDDEN_WIDGETS = ["pdd_file"];
const TURBO_STEPS_WIDGETS = ["pdd_nfe"];
const HYPERFLOW_ONLY_WIDGETS = ["hyperflow_file", "hyperflow_curve_refit", "hyperflow_strength", "hyperflow_lora_mode"];
const SLA_ONLY_WIDGETS = ["sla_sparsity", "pass2_sla_sparsity"];
// Dual sampling (appended last, after sparse_sink): HyperFlow only; the split step and early stop only while it is on.
const DUAL_WIDGET = "dual_sampling";
const DUAL_DETAIL_WIDGETS = ["dual_split_step", "pass1_early_stop"];
// pass1_steps (appended after pass1_early_stop): HyperFlow without dual sampling only; 0 = the full 8-step pass 1.
const PASS1_STEPS_WIDGET = "pass1_steps";
// The engine combo value that is HyperFlow with dual sampling forced on (the dual_sampling toggle is then redundant).
const DUAL_MODE = "HyperFlow 8-step \u00b7 dual sampling";
const isDualMode = (value) => String(value ?? "") === DUAL_MODE;
// Widgets appended after master_ui-less workflows were saved: a missing/invalid value is put back to the default.
const AUDIO_REFINE_WIDGETS = ["audio_refine_steps", "audio_refine_denoise", "audio_refine_cache"];
// auto_clips is appended after rewrite_story; a workflow saved before it carries the empty master_ui value there.
const AUTO_CLIPS_WIDGETS = ["auto_clips", "planner_refs", "auto_clip_seconds"];
// story_engine and its E4 settings are appended after auto_clip_seconds, so the same empty master_ui value lands in story_engine.
// Only the dropdowns are repaired: the two text widgets (e4_language, e4_picture_notes) read an empty value as their default.
const STORY_ENGINE_WIDGETS = ["story_engine", "e4_score", "e4_decision_budget"];
const REPAIRED_WIDGETS = [...HYPERFLOW_ONLY_WIDGETS, ...AUDIO_REFINE_WIDGETS, ...AUTO_CLIPS_WIDGETS, ...STORY_ENGINE_WIDGETS];

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (widget._origType === undefined) {
        widget._origType = widget.type;
        widget._origComputeSize = widget.computeSize;
    }
    if (visible) {
        widget.type = widget._origType;
        widget.computeSize = widget._origComputeSize;
    } else {
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
    }
}

// Workflows saved before the HyperFlow inputs existed stored the DOM widget master_ui's empty
// string in the slot that hyperflow_file now occupies, so it loads as "" and the server rejects
// it (value_not_in_list). Put any missing/invalid HyperFlow value back to the node-definition default.
// The same happens one slot later for audio_refine_steps in workflows saved before the audio-refine inputs,
// and for final_decode (the last native widget, just before master_ui) in workflows saved before it existed.
function hyperflowValueOk(widget, value) {
    if (widget.type === "combo" || widget._origType === "combo") {
        const values = comboValues(widget);
        return typeof value === "string" && value !== "" && (!values.length || values.includes(value));
    }
    if (widget._origType === "toggle" || widget.type === "toggle") return typeof value === "boolean";
    return typeof value === "number" && Number.isFinite(value);
}

// final_decode is "standard" or a packed 2x VAE file. "" (the old master_ui slot) -> standard;
// the first release's "X2 detail" -> the first detected 2x VAE, or standard when none is installed.
function repairFinalDecode(node) {
    const w = node.widgets?.find(x => x.name === "final_decode");
    if (!w) return;
    const values = comboValues(w);
    if (typeof w.value !== "string" || w.value === "") w.value = "standard";
    else if (w.value === "X2 detail") w.value = values.find(v => v !== "standard") ?? "standard";
}

// rewrite_reasoning_budget is a 1024/2048/4096 dropdown now; saved workflows and projects carry the old INT.
function repairReasoningBudget(node) {
    const w = node.widgets?.find(x => x.name === "rewrite_reasoning_budget");
    if (w) w.value = normalizeReasoningBudget(w.value);
}

// rewrite_caption_model was merged into rewrite_writer_model. LiteGraph has already put the old array's values
// into the widgets by index, so re-seat them from the array with the caption model's value removed.
function repairMergedRewriterModel(node, info) {
    const names = (node.widgets || []).map(w => w.name);
    const values = info?.widgets_values;
    const afterCaption = dropCaptionModelValue(names, values);
    const afterSingle = dropSingleTextEncodeValue(names, afterCaption ?? values);
    const afterAccel = migrateAccelMode(names, afterSingle ?? afterCaption ?? values);
    const migrated = afterAccel ?? afterSingle ?? afterCaption;
    if (!migrated) return;
    info.widgets_values = migrated;
    migrated.forEach((value, i) => { if (node.widgets[i]) node.widgets[i].value = value; });
}

// sparse_start_percent / sparse_dense_blocks / sparse_sink are the last widgets, after e4_picture_notes; a workflow
// saved before them (including every E4-era one) has its trailing master_ui "" land in the first of them. Anything not a number there means auto.
function repairSparseWidgets(node) {
    const w = node.widgets?.find(x => x.name === "sparse_start_percent");
    if (w && (typeof w.value !== "number" || !Number.isFinite(w.value))) w.value = -1;
}

// BUNNY ActionLogic Bridge V1 was deleted from the box (2026-10-07); saved workflows that pick it load on V2.
const RETIRED_BRIDGES = { "BUNNY_H3_ActionLogic_Bridge_V1.safetensors": "BUNNY_H3_ActionLogic_Bridge_V2.safetensors" };
function repairSemanticBridge(node) {
    const w = node.widgets?.find(x => x.name === "semantic_bridge");
    const replacement = w && RETIRED_BRIDGES[w.value];
    if (replacement && comboValues(w).includes(replacement)) w.value = replacement;
}

// The first E4 builds named the engine "e4", the next "e4.8"; both are "e4.10" now (the vendored engine is E4.10).
function repairStoryEngine(node) {
    const w = node.widgets?.find(x => x.name === "story_engine");
    if (w && (w.value === "e4" || w.value === "e4.8")) w.value = "e4.10";
}

function repairDualWidgets(node) {
    for (const name of [DUAL_WIDGET, ...DUAL_DETAIL_WIDGETS]) {
        const w = node.widgets?.find(x => x.name === name);
        if (w) w.value = dualWidgetValue(name, w.value);
    }
    const steps = node.widgets?.find(x => x.name === PASS1_STEPS_WIDGET);
    if (steps) steps.value = pass1StepsValue(steps.value);
    const chunkWhen = node.widgets?.find(x => x.name === "pass2_chunk_when");
    if (chunkWhen) chunkWhen.value = pass2ChunkWhenValue(chunkWhen.value);
    const p2Sla = node.widgets?.find(x => x.name === "pass2_sla_sparsity");
    if (p2Sla) p2Sla.value = pass2SlaSparsityValue(p2Sla.value);
}

function repairHyperflowWidgets(node, defaults) {
    repairDualWidgets(node);
    repairFinalDecode(node);
    repairStoryEngine(node);
    repairReasoningBudget(node);
    for (const name of REPAIRED_WIDGETS) {
        const w = node.widgets?.find(x => x.name === name);
        if (!w || hyperflowValueOk(w, w.value)) continue;
        let fallback = defaults?.[name];
        if (w.type === "combo" || w._origType === "combo") {
            const values = comboValues(w);
            if (values.length && !values.includes(fallback)) fallback = values[0];
        }
        if (fallback !== undefined) w.value = fallback;
    }
}

function setupModeWidgets(node) {
    const byName = (name) => node.widgets?.find(w => w.name === name);
    const accel = byName("accel_mode");
    const sla = byName("sla_enabled");
    if (!accel && !sla) return;

    const apply = () => {
        const mode = String(accel?.value ?? "").toLowerCase();
        const turbo = mode.startsWith("turbo");
        const hyperflow = !turbo;
        const slaOn = sla ? !!sla.value : false;
        TURBO_ONLY_WIDGETS.forEach(n => setWidgetVisible(byName(n), turbo));
        HYPERFLOW_ONLY_WIDGETS.forEach(n => setWidgetVisible(byName(n), hyperflow));
        const dualMode = isDualMode(accel?.value);
        const dualOn = hyperflow && (dualMode || !!byName(DUAL_WIDGET)?.value);
        setWidgetVisible(byName(DUAL_WIDGET), hyperflow && !dualMode);
        DUAL_DETAIL_WIDGETS.forEach(n => setWidgetVisible(byName(n), dualOn));
        setWidgetVisible(byName(PASS1_STEPS_WIDGET), hyperflow && !dualOn);
        ALWAYS_HIDDEN_WIDGETS.forEach(n => setWidgetVisible(byName(n), false));
        TURBO_STEPS_WIDGETS.forEach(n => setWidgetVisible(byName(n), turbo));
        SLA_ONLY_WIDGETS.forEach(n => setWidgetVisible(byName(n), slaOn));
        // Let LiteGraph recompute the node height without shrinking the DOM
        // editor that sits at the bottom of the node.
        const size = node.computeSize();
        node.setSize([Math.max(node.size[0], size[0]), Math.max(node.size[1], size[1])]);
        node.setDirtyCanvas(true, true);
    };
    node.applyModeWidgets = apply;

    for (const w of [accel, sla, byName(DUAL_WIDGET)]) {
        if (!w) continue;
        const original = w.callback;
        w.callback = function () {
            const r = original?.apply(this, arguments);
            apply();
            return r;
        };
    }
    apply();
}

// ---------------------------------------------------------------------------
// Simplified panel: every native widget is hidden and drawn by the DOM panel
// instead, grouped into Engine / Quality / Continuity / Performance / Clips.
// Values still live in the native widgets, so the API and saved workflows are
// unchanged; this is presentation only.
// ---------------------------------------------------------------------------
const ORIENTATIONS = ["16:9", "9:16", "1:1"];
const QUALITY_PRESETS = {
    draft:    { label: "Draft · 608p",    pass1: { "16:9": "608x352 (16:9)", "9:16": "352x608 (9:16)", "1:1": "512x512 (1:1)" },
                pass2: { "16:9": "1056x608 (16:9)", "9:16": "608x1056 (9:16)", "1:1": "1024x1024 (1:1)" }, denoise: 0.25 },
    standard: { label: "Standard · 720p", pass1: { "16:9": "608x352 (16:9)", "9:16": "352x608 (9:16)", "1:1": "512x512 (1:1)" },
                pass2: { "16:9": "1280x720", "9:16": "720x1280 (9:16)", "1:1": "1024x1024 (1:1)" }, denoise: 0.25 },
    high:     { label: "High · 768p",     pass1: { "16:9": "608x352 (16:9)", "9:16": "352x608 (9:16)", "1:1": "512x512 (1:1)" },
                pass2: { "16:9": "1344x768 (16:9)", "9:16": "768x1344 (9:16)", "1:1": "1024x1024 (1:1)" }, denoise: 0.25 },
    // 1080p drafts at 608p: the 3D latent upscaler is a ~2x model, and a 352p draft
    // blown up 3x leaves it inventing detail the refine pass then has to fix.
    cinema:   { label: "Cinema · 1080p",  pass1: { "16:9": "1056x608 (16:9)", "9:16": "608x1056 (9:16)", "1:1": "512x512 (1:1)" },
                pass2: { "16:9": "1920x1088 (16:9)", "9:16": "1088x1920 (9:16)", "1:1": "1024x1024 (1:1)" }, denoise: 0.25 },
};

function orientationOf(resolution) {
    const r = String(resolution || "");
    if (r.includes("9:16")) return "9:16";
    if (r.includes("1:1")) return "1:1";
    const m = /^(\d+)x(\d+)/.exec(r);
    if (m) { const w = +m[1], h = +m[2]; if (w === h) return "1:1"; if (h > w) return "9:16"; }
    return "16:9";
}

function detectPreset(pass1, pass2, denoise) {
    for (const [key, preset] of Object.entries(QUALITY_PRESETS)) {
        for (const o of ORIENTATIONS) {
            if (preset.pass1[o] === pass1 && preset.pass2[o] === pass2 && Math.abs(Number(denoise) - preset.denoise) < 1e-6) {
                return { preset: key, orientation: o };
            }
        }
    }
    return { preset: "custom", orientation: orientationOf(pass2) };
}

function comboValues(widget) {
    const v = widget?.options?.values;
    return Array.isArray(v) ? v : (typeof v === "function" ? (v() || []) : []);
}

function shortModelName(name) {
    return String(name || "none").replace(/\.(safetensors|pt|pth|ckpt)$/i, "").replace(/^minimax[_-]?h3[_-]?/i, "");
}

function hideNativeWidgets(node) {
    for (const w of node.widgets || []) {
        if (w.name === "master_ui") continue;
        setWidgetVisible(w, false);
    }
    node.setDirtyCanvas(true, true);
}

const UI_ROW_CSS = "display: flex; align-items: center; justify-content: space-between; gap: 10px; min-height: 24px; padding: 2px 10px; background: #1c1c28; border: 1px solid #2a2a3c; border-radius: 12px; font-size: 11.5px; color: #b9b9cf;";
const UI_CONTROL_CSS = "background: #101016; border: 1px solid #2e2e42; border-radius: 4px; color: #ececf4; font-size: 11.5px; padding: 2px 6px; max-width: 62%;";

function uiRow(label, control, { hint, indent } = {}) {
    const row = document.createElement("div");
    row.style.cssText = UI_ROW_CSS + (indent ? " margin-left: 14px; border-left: 2px solid #3c3c56; border-radius: 0 12px 12px 0;" : "");
    const left = document.createElement("span");
    left.textContent = label;
    if (hint) left.title = hint;
    row.appendChild(left);
    row.appendChild(control);
    return row;
}

function uiSelect(values, current, onChange, { render } = {}) {
    const sel = document.createElement("select");
    sel.style.cssText = UI_CONTROL_CSS;
    for (const v of values) {
        const opt = document.createElement("option");
        opt.value = String(v);
        opt.textContent = render ? render(v) : String(v);
        if (String(v) === String(current)) opt.selected = true;
        sel.appendChild(opt);
    }
    sel.onchange = () => onChange(sel.value);
    return sel;
}

function uiNumber(current, onChange, { min, max, step } = {}) {
    const inp = document.createElement("input");
    inp.type = "number";
    inp.style.cssText = UI_CONTROL_CSS + " width: 90px; text-align: right;";
    if (min !== undefined) inp.min = min;
    if (max !== undefined) inp.max = max;
    if (step !== undefined) inp.step = step;
    inp.value = current;
    inp.onchange = () => onChange(Number(inp.value));
    return inp;
}

function uiText(current, onChange, { width = 90 } = {}) {
    const inp = document.createElement("input");
    inp.type = "text";
    inp.style.cssText = UI_CONTROL_CSS + ` width: ${width}px; text-align: right;`;
    inp.value = current;
    inp.onchange = () => onChange(inp.value);
    return inp;
}

function uiToggle(current, onChange) {
    const label = document.createElement("label");
    label.style.cssText = "display: inline-flex; align-items: center; cursor: pointer;";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = Boolean(current);
    box.style.cssText = "accent-color: #6355d8; width: 15px; height: 15px; cursor: pointer;";
    box.onchange = () => onChange(box.checked);
    label.appendChild(box);
    return label;
}

function uiTextarea(current, onChange, { rows = 4, placeholder } = {}) {
    const ta = document.createElement("textarea");
    ta.style.cssText = "width: 100%; box-sizing: border-box; background: #101016; border: 1px solid #2e2e42; border-radius: 4px; color: #ececf4; font-size: 11px; padding: 4px 6px; resize: vertical; font-family: inherit;";
    ta.rows = rows;
    ta.spellcheck = false;
    if (placeholder) ta.placeholder = placeholder;
    ta.value = current ?? "";
    let timer = 0;
    ta.oninput = () => { clearTimeout(timer); timer = setTimeout(() => onChange(ta.value), 200); };
    ta.addEventListener("keydown", (e) => e.stopPropagation());
    ta.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });
    ta.addEventListener("pointerdown", (e) => e.stopPropagation());
    return ta;
}

function modelLabel(v) {
    // The rewriter pack's labels carry size and note after " · "; the name is enough here.
    return String(v ?? "").split(" · ")[0];
}

function uiSegmented(values, current, onChange) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display: flex; background: #101016; border: 1px solid #2e2e42; border-radius: 12px; padding: 2px; gap: 2px;";
    for (const v of values) {
        const b = document.createElement("button");
        b.textContent = v;
        const on = v === current;
        b.style.cssText = `border: none; cursor: pointer; padding: 2px 10px; border-radius: 9px; font-size: 11px; background: ${on ? "#6355d8" : "transparent"}; color: ${on ? "#fff" : "#9a9ab4"};`;
        b.onclick = () => onChange(v);
        wrap.appendChild(b);
    }
    return wrap;
}

function uiSectionHeader(title, { summary, open, collapsible, onToggle, actions } = {}) {
    const head = document.createElement("div");
    head.style.cssText = `display: flex; align-items: center; justify-content: space-between; gap: 8px; padding: 4px 10px; border-radius: 6px; background: #1a1a26; border: 1px solid #262636; ${collapsible ? "cursor: pointer;" : ""} user-select: none;`;
    const left = document.createElement("span");
    left.style.cssText = "font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #9d8cff;";
    left.textContent = title;
    const right = document.createElement("span");
    right.style.cssText = "display: flex; align-items: center; gap: 8px; font-size: 11px; color: #7f7f9a; min-width: 0;";
    if (summary) {
        const sum = document.createElement("span");
        sum.style.cssText = "overflow: hidden; text-overflow: ellipsis; white-space: nowrap;";
        sum.textContent = summary;
        right.appendChild(sum);
    }
    if (actions) right.appendChild(actions);
    if (collapsible) {
        const chev = document.createElement("span");
        chev.textContent = open ? "▴" : "▾";
        chev.style.cssText = "color: #9a9ab4; font-size: 12px;";
        right.appendChild(chev);
        head.onclick = () => onToggle?.();
    }
    head.appendChild(left);
    head.appendChild(right);
    return head;
}

function uiHint(text) {
    const d = document.createElement("div");
    d.style.cssText = "font-size: 10.5px; color: #6f6f8a; padding: 0 4px; line-height: 1.35;";
    d.textContent = text;
    return d;
}

app.registerExtension({
    name: "MiniMaxH3.MasterExtender",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== TARGET_NODE) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            this.setupMasterUI();
            return r;
        };

        const hyperflowDefaults = {};
        for (const name of REPAIRED_WIDGETS) {
            const spec = nodeData.input?.optional?.[name] ?? nodeData.input?.required?.[name];
            if (spec?.[1]?.default !== undefined) hyperflowDefaults[name] = spec[1].default;
        }

        nodeType.prototype.setupMasterUI = function () {
            const node = this;
            node.setSize([960, 680]);

            // Hide raw JSON widgets
            const clipsWidget = node.widgets?.find(w => w.name === "clips_json");
            const refsWidget = node.widgets?.find(w => w.name === "refs_json");

            if (clipsWidget) clipsWidget.type = "hidden";
            if (refsWidget) refsWidget.type = "hidden";

            let clipsState = [];
            // Validated clips show as one line; ids the user expanded this session.
            const expandedValidated = new Set();
            // Disk-cache state per clip index, from /minimax_master/cache_status.
            let cacheStates = [];
            let cacheVideos = [];
            let cacheTimer = 0;
            try {
                clipsState = JSON.parse(clipsWidget.value || "[]");
            } catch (e) {
                clipsState = [];
            }
            // The side panel addresses clips by id, so ids must be unique numbers.
            const nextClipId = (clips) => clips.reduce((m, c) => Math.max(m, typeof c.id === "number" ? c.id : -1), -1) + 1;
            const ensureClipIds = (clips) => {
                const seen = new Set();
                clips.forEach((c) => {
                    if (typeof c.id !== "number" || seen.has(c.id)) c.id = nextClipId(clips);
                    seen.add(c.id);
                });
                return clips;
            };
            if (Array.isArray(clipsState)) ensureClipIds(clipsState);
            if (!Array.isArray(clipsState) || clipsState.length === 0) {
                clipsState = [
                    {
                        id: 0,
                        title: "Clip 1",
                        prompt: "",
                        duration: DUR_DEFAULT,
                        seed: Math.floor(Math.random() * 1000000000),
                        seed_mode: "randomize",
                        validated: false,
                        loras: []
                    }
                ];
            }

            // Create Master UI DOM container
            const container = document.createElement("div");
            container.className = "minimax-master-container";
            container.style.cssText = `
                display: flex;
                flex-direction: column;
                gap: 12px;
                padding: 12px;
                background: #14141c;
                border-radius: 8px;
                border: 1px solid #2a2a3c;
                color: #e2e2ec;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 12px;
                user-select: none;
                max-width: 930px;
            `;

            function saveState() {
                if (clipsWidget) {
                    clipsWidget.value = JSON.stringify(clipsState, null, 2);
                }
                node.setDirtyCanvas(true, true);
                refreshCacheStatus();
            }

            function refreshCacheStatus() {
                clearTimeout(cacheTimer);
                cacheTimer = setTimeout(async () => {
                    try {
                        const response = await api.fetchApi("/minimax_master/cache_status", {
                            method: "POST", headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({
                                clips: JSON.stringify(clipsState),
                                audio_refine: [getW("audio_refine_steps", 0), getW("audio_refine_denoise", 0.5), getW("audio_refine_cache", true)],
                            }),
                        });
                        const result = await response.json();
                        const states = result.states || [], videos = result.videos || [];
                        if (JSON.stringify([states, videos]) !== JSON.stringify([cacheStates, cacheVideos])) {
                            cacheStates = states;
                            cacheVideos = videos;
                            renderUI();
                        }
                    } catch (error) {
                        console.warn("MiniMax Master: cache status check failed", error);
                    }
                }, 400);
            }

            const CACHE_BADGES = {
                cached: ["● cached", "#4ade80", "On disk, rendered from exactly this prompt, seed, duration and LoRAs (and the clips before it). Validating it reuses the render."],
                stale: ["● changed", "#f59e0b", "On disk, but the prompt, seed, duration, LoRAs or an earlier clip changed since it was rendered. It will render again."],
                unverified: ["● on disk", "#9ca3af", "On disk from before cache checks existed; cannot confirm it matches the current inputs."],
                none: ["○ not cached", "#6b7280", "Not rendered yet for these inputs."],
            };
            function cacheBadge(index) {
                const badge = CACHE_BADGES[cacheStates[index]];
                return badge ? `<span title="${badge[2]}" style="color: ${badge[1]}; font-size: 10px; white-space: nowrap;">${badge[0]}</span>` : "";
            }

            function clipVideo(index, style) {
                const info = cacheVideos[index];
                if (!info) return null;
                const video = document.createElement("video");
                const query = new URLSearchParams({ filename: info.filename, subfolder: info.subfolder, type: "output", t: String(info.mtime) });
                video.src = api.apiURL(`/view?${query}`);
                video.muted = true;
                video.loop = true;
                video.playsInline = true;
                video.preload = "metadata";
                video.title = "Rendered clip (hover to play)";
                video.style.cssText = `display: block; background: #000; border-radius: 4px; object-fit: contain; ${style}`;
                video.onmouseenter = () => video.play().catch(() => {});
                video.onmouseleave = () => video.pause();
                return video;
            }

            function copyButton(clip) {
                const button = document.createElement("button");
                button.textContent = "⧉ Copy";
                button.title = "Copy the prompt this clip renders from (the rewritten text when it was rewritten)";
                button.style.cssText = "background: #252536; border: 1px solid #3f3f58; color: #ddd; border-radius: 3px; cursor: pointer; padding: 1px 6px; font-size: 10px; white-space: nowrap;";
                button.onclick = async (e) => {
                    e.stopPropagation();
                    try {
                        await copyText(clip.prompt || "");
                        button.textContent = "✓ Copied";
                    } catch (error) {
                        button.textContent = "Copy failed";
                    }
                    setTimeout(() => { button.textContent = "⧉ Copy"; }, 1200);
                };
                return button;
            }

            // An API-format graph (a video whose only tag is the queued prompt) never runs onConfigure: the loader sets
            // clips_json after the node exists and then calls the widget's callback. Take the clips from there too.
            const clipsCallback = clipsWidget.callback;
            clipsWidget.callback = function (value) {
                const result = clipsCallback?.apply(this, arguments);
                try {
                    const restored = JSON.parse(value || "[]");
                    if (Array.isArray(restored)) {
                        clipsState = ensureClipIds(restored);
                        renderUI();
                    }
                } catch (error) {
                    console.warn("MiniMax Master: invalid clips JSON", error);
                }
                return result;
            };

            const originalConfigure = node.onConfigure;
            node.onConfigure = function (info) {
                const result = originalConfigure?.apply(this, arguments);
                repairMergedRewriterModel(node, info);
                repairHyperflowWidgets(node, hyperflowDefaults);
                repairSemanticBridge(node);
                repairSparseWidgets(node);
                // Older workflows stored the empty master_ui value in this slot.
                const attentionWidget = node.widgets?.find(w => w.name === "attention_backend");
                if (attentionWidget && attentionWidget.value === "") {
                    attentionWidget.value = "comfy kitchen attention";
                }
                try {
                    const restored = JSON.parse(clipsWidget.value || "[]");
                    if (Array.isArray(restored)) clipsState = ensureClipIds(restored);
                } catch (error) {
                    console.warn("MiniMax Master: invalid saved clips JSON", error);
                }
                // configure() replaced node.properties.h3_ui with the saved object; take its
                // panel state (mode etc.) into the live `ui` and point the property back at it,
                // but start with every section collapsed so the Clips section fits.
                const savedUi = node.properties?.h3_ui;
                if (savedUi && savedUi !== ui && typeof savedUi === "object") {
                    for (const [key, value] of Object.entries(savedUi)) if (key !== "open") ui[key] = value;
                }
                ui.open = { engine: false, quality: false, continuity: false, performance: false, rewriter: false };
                node.properties.h3_ui = ui;
                renderUI();
                hideNativeWidgets(node);
                return result;
            };

            const projectControls = createProjectControls(node, {
                getClips: () => clipsState,
                setClips: clips => { clipsState = clips; },
                save: saveState,
                render: () => renderUI(),
            });
            const restoreProject = node.masterProjectRestore;
            node.masterProjectRestore = () => { repairReasoningBudget(node); restoreProject?.(); };

            // ---- Simplified panel state (persisted in node.properties) ----
            node.properties = node.properties || {};
            const ui = node.properties.h3_ui = Object.assign(
                { mode: "simple", open: { engine: true, quality: true, continuity: false, performance: false, rewriter: false } },
                node.properties.h3_ui || {},
            );
            // Every settings section starts collapsed when the workflow loads (each shows a
            // one-line summary), so the node's default size reaches the Clips section; the
            // open state saved in the workflow is deliberately not restored.
            ui.open = { engine: false, quality: false, continuity: false, performance: false, rewriter: false };
            const expert = () => ui.mode === "expert";

            const W = (name) => node.widgets?.find((w) => w.name === name);
            const getW = (name, fallback) => { const w = W(name); return w ? w.value : fallback; };
            const setW = (name, value, { rerender = true } = {}) => {
                const w = W(name);
                if (!w) return;
                w.value = value;
                w.callback?.(value, app.canvas, node);
                node.setDirtyCanvas(true, true);
                app.graph?.setDirtyCanvas?.(true, true);
                if (rerender) renderUI();
            };
            const isTurbo = () => String(getW("accel_mode", "")).toLowerCase().startsWith("turbo");
            const isHyperflow = () => !isTurbo();
            const isOpenH3 = () => String(getW("pass2_refiner", "")).toLowerCase().startsWith("open-h3");

            function bindSelect(name, opts = {}) {
                const w = W(name);
                const values = comboValues(w);
                return uiSelect(values.length ? values : [w?.value ?? ""], w?.value, (v) => setW(name, v, opts), { render: opts.render });
            }
            function bindNumber(name, opts = {}) {
                const w = W(name);
                const o = w?.options || {};
                // LiteGraph stores FLOAT steps x10 in options.step; step2/round carry the real increment.
                const step = o.step2 ?? o.round ?? (typeof o.step === "number" ? o.step / 10 : "any");
                return uiNumber(w?.value ?? 0, (v) => setW(name, v, opts), { min: o.min, max: o.max, step });
            }
            function bindToggle(name, opts = {}) {
                return uiToggle(getW(name, false), (v) => setW(name, v, opts));
            }

            // "Custom" is an explicit choice, remembered in ui.qualityCustom; otherwise the
            // preset is derived from the widget values so saved workflows land on their tier.
            function currentPreset() {
                const detected = detectPreset(getW("pass1_resolution"), getW("pass2_resolution"), getW("pass2_denoise"));
                return ui.qualityCustom ? { preset: "custom", orientation: detected.orientation } : detected;
            }

            function applyQualityPreset(key, orientation) {
                const preset = QUALITY_PRESETS[key];
                if (!preset) return;
                ui.qualityCustom = false;
                const p1 = comboValues(W("pass1_resolution"));
                const p2 = comboValues(W("pass2_resolution"));
                const want1 = preset.pass1[orientation], want2 = preset.pass2[orientation];
                if (!p1.includes(want1) || !p2.includes(want2)) {
                    console.warn("MiniMax Master: preset resolutions not offered by this node build", want1, want2);
                    return;
                }
                setW("pass1_resolution", want1, { rerender: false });
                setW("pass2_resolution", want2, { rerender: false });
                setW("pass2_denoise", preset.denoise, { rerender: false });
                renderUI();
            }

            function engineSummary() {
                const steps = getW("pdd_nfe", "8");
                const p2 = String(getW("pass2_lora", "none"));
                const p2note = isOpenH3() ? " · refine Open-H3 + face"
                    : p2 !== "none" ? ` · refine ${shortModelName(p2)}${String(getW("pass2_lora_mode", "")).startsWith("replace") ? " (replace)" : ""}` : "";
                const bridge = String(getW("semantic_bridge", "none"));
                const bridgeNote = bridge !== "none" ? ` · bridge α ${Number(getW("semantic_bridge_alpha", 0)).toFixed(2)}` : "";
                const dualMode = isDualMode(getW("accel_mode", ""));
                const dualNote = (dualMode || getW(DUAL_WIDGET, false))
                    ? ` · dual sampling (pass 2 from step ${getW("dual_split_step", 2)}${dualMode ? ", whole clip" : ""}${getW("pass1_early_stop", false) ? ", pass 1 early stop" : ""})` : "";
                const p1s = Number(getW(PASS1_STEPS_WIDGET, 0));
                const p1sNote = (isHyperflow() && !dualNote && p1s > 0) ? ` · pass 1 stops at step ${p1s}/8` : "";
                if (dualMode) return `HyperFlow · dual sampling · ${shortModelName(getW("hyperflow_file"))} · refit ${getW("hyperflow_curve_refit", true) ? "on" : "off"} · euler${dualNote.replace(" · dual sampling", "")}${bridgeNote}`;
                if (isHyperflow()) return `HyperFlow 8-step · ${shortModelName(getW("hyperflow_file"))} · refit ${getW("hyperflow_curve_refit", true) ? "on" : "off"} · euler${p2note}${dualNote}${p1sNote}${bridgeNote}`;
                return `Turbo LoRA · ${shortModelName(getW("turbo_lora"))} · ${steps} steps · ${getW("turbo_sampler", "")} / ${getW("turbo_scheduler", "")}${p2note}${bridgeNote}`;
            }
            function qualitySummary() {
                const { preset, orientation } = currentPreset();
                const label = preset === "custom" ? "Custom" : QUALITY_PRESETS[preset].label;
                return `${label} · ${orientation} · draft ${String(getW("pass1_resolution", "")).split(" ")[0]} → refine ${String(getW("pass2_resolution", "")).split(" ")[0]}, denoise ${Number(getW("pass2_denoise", 0)).toFixed(2)}${getW("final_decode", "standard") !== "standard" ? " · 2x VAE" : ""}`;
            }
            function performanceSummary() {
                const att = String(getW("attention_backend", "")).replace(" attention", "");
                const sla = getW("sla_enabled", false) ? `SLA ${Number(getW("sla_sparsity", 0)).toFixed(2)} ${getW("sparse_method", "sla")}` : "SLA off";
                const cf = Number(getW("pass2_chunk_frames", 0));
                const chunk = cf > 0 ? `chunk ${cf}/${getW("pass2_chunk_overlap", 0)}${getW("pass2_chunk_when", "above 720p") === "always" ? " always" : " >720p"}` : "no chunking";
                return `${att} · ${sla} · ${chunk}${getW("smart_offload", true) ? "" : " · offload off"}${getW("async_decode", "off") !== "off" ? ` · async ${getW("async_decode")}` : ""}`;
            }
            function continuitySummary() {
                return `motion ${getW("context_length", "22")} f · audio ${getW("audio_context_length", 0)} f · identity ${getW("identity_continuity", true) ? "on" : "off"}`;
            }

            function rewriterSummary() {
                const mode = String(getW("rewrite_mode", "off"));
                if (!W("rewrite_mode")) return "not available";
                if (mode === "off") return "off";
                const parts = [mode, modelLabel(getW("rewrite_writer_model", ""))];
                if (getW("rewrite_thinking", false)) parts.push(`thinking ${getW("rewrite_reasoning_budget", 0)}`);
                return parts.join(" · ");
            }

            function buildRewriter(wrap) {
                if (!W("rewrite_mode")) {
                    wrap.appendChild(uiHint("This node build has no built-in rewriter widgets; update the Master Extender pack."));
                    return;
                }
                wrap.appendChild(uiRow("rewrite prompts", bindSelect("rewrite_mode"), {
                    hint: "pending clips: rewrite clips not rewritten yet, at the start of the run. all clips: rewrite every clip from its raw text again.",
                }));
                if (String(getW("rewrite_mode", "off")) === "off") {
                    wrap.appendChild(uiHint("Off: clip prompts are used exactly as typed (or as fed by the clip_prompt inputs)."));
                    return;
                }
                const noPack = String(getW("rewrite_writer_model", "")).startsWith("(");
                if (noPack) wrap.appendChild(uiHint("The MiniMax-H3-Prompt-Rewriter-ComfyUI pack is not installed beside this node; the run will stop with that message."));
                wrap.appendChild(uiRow("task", bindSelect("rewrite_task"), { hint: "auto: Ref2VA when reference pictures are connected, else T2VA." }));
                wrap.appendChild(uiRow("rewriter model", bindSelect("rewrite_writer_model", { render: modelLabel }), { hint: "One model looks at the reference pictures and writes the prompts, in one call per clip, on one server session. A model with vision (an mmproj, or Strata with its vision section) is given the pictures themselves, with no caption step, and thinking is available. A model without vision (a plain GGUF) writes text-only and the pictures are only labelled <Picture N>." }));
                wrap.appendChild(uiRow("caption length", bindSelect("rewrite_caption_length"), { indent: true }));
                wrap.appendChild(uiRow("thinking", bindToggle("rewrite_thinking")));
                if (getW("rewrite_thinking", false)) {
                    wrap.appendChild(uiRow("thinking budget (tokens)", bindSelect("rewrite_reasoning_budget"), { indent: true, hint: "Thinking tokens at most, on top of the answer length." }));
                    const msg = uiTextarea(getW("rewrite_reasoning_budget_message", ""), (v) => setW("rewrite_reasoning_budget_message", v, { rerender: false }), { rows: 2, placeholder: "Message injected when the budget runs out, e.g. 'Time is up, write the answer now.'" });
                    const msgWrap = document.createElement("div");
                    msgWrap.style.cssText = "margin-left: 14px;";
                    msgWrap.appendChild(msg);
                    wrap.appendChild(msgWrap);
                }
                if (W("rewrite_previous_clips")) wrap.appendChild(uiRow("continuity for clip N", bindSelect("rewrite_previous_clips"), { hint: "raw asks: earlier clips' asks as you typed them (written in parallel). final prompts: clip N-1's full rewritten prompt plus the earlier raw asks, so clip N continues from what N-1 became (written one after another; N is redone when N-1's final prompt changes). off: nothing." }));
                if (W("rewrite_story")) {
                    wrap.appendChild(uiHint("Film story (optional): the whole story, start to end. Every clip's writer gets it and picks only the next beat after the previous clips. Changing it does not redo clips already rewritten."));
                    const storyBox = document.createElement("div");
                    storyBox.style.cssText = "margin-left: 14px;";
                    storyBox.appendChild(uiTextarea(getW("rewrite_story", ""), (v) => setW("rewrite_story", v, { rerender: false }), { rows: 4, placeholder: "The whole film in a few lines: who, where, how it begins, turns and ends. Each clip then covers only its own beat." }));
                    wrap.appendChild(storyBox);
                    if (W("auto_clips")) {
                        wrap.appendChild(uiRow("plan story into N clips", bindNumber("auto_clips"), { hint: "0 = off. With a film story set, the first run plans it ONCE into exactly N clips (length below, 15 s by default; shot lists written into the empty clips, flagged planned, editable), then rewrites them. Clips you typed are kept and planned around: the planner sees them and fills only the empty ones. Changing the story never replans. Raise to plan more clips after the existing ones (only the new ones are planned, continuing from the last clip); lowering does nothing." }));
                        if (Number(getW("auto_clips", 0)) > 0) {
                            wrap.appendChild(uiHint("Raise the number to plan more clips after the existing ones (only the new clips are planned)."));
                            if (W("auto_clip_seconds")) wrap.appendChild(uiRow("planned clip length (s)", bindNumber("auto_clip_seconds"), { indent: true, hint: "Seconds per planned clip, 5-15 (15 = the old fixed length). The planner budgets its beats and shots to this length and the planned clips are created with it. Typed clips keep their own duration. Applies only when a plan is made; to re-plan at a new length use 'Replan from story'." }));
                            if (W("planner_refs")) wrap.appendChild(uiRow("planner sees references as", bindSelect("planner_refs"), { indent: true, hint: "images: the reference pictures themselves, as the clip writers get them (needs a rewriter model with vision, else it is the story text alone); captions: the old name for images, now the same; off: the story text alone." }));
                            if (W("story_engine")) {
                                wrap.appendChild(uiRow("story engine", bindSelect("story_engine"), { indent: true, hint: "builder: planner.md plans, builder.md writes every clip (as before). e4.10 / e4.11: the E4.10 or E4.11.1 pipeline plans the whole film and writes every clip's final prompt (film bible, per-clip staging decisions, speaker ids, every spoken word verbatim, checked and repaired). e4.11 is E4.10 built for few repair calls (fixes in code first) and parallel requests (up to rewrite_parallel, at most 4). Both need Node.js 18+, a server-backed rewriter, at least one reference picture and an empty clip list; it decides the number of clips itself and plans 15 s clips." }));
                                if (["e4.10", "e4.11"].includes(String(getW("story_engine", "builder")))) {
                                    const lang = document.createElement("div");
                                    lang.style.cssText = "margin-left: 28px;";
                                    lang.appendChild(uiTextarea(getW("e4_language", "English"), (v) => setW("e4_language", v, { rerender: false }), { rows: 1, placeholder: "Dialogue language, e.g. English" }));
                                    if (W("e4_language")) wrap.appendChild(uiRow("dialogue language", lang, { indent: true, hint: "The language every spoken line is written in." }));
                                    if (W("e4_score")) wrap.appendChild(uiRow("score", bindSelect("e4_score"), { indent: true, hint: "off: non_diegetic_music is N/A and music-like sound words are swept out. on: one instrumental score for the film." }));
                                    if (W("e4_decision_budget")) wrap.appendChild(uiRow("decision thinking", bindSelect("e4_decision_budget"), { indent: true, hint: "Thinking of each per-clip decision call. off (default): no thinking, measured 0.6 points below thinking 2048 (not significant) and about 4.7 times faster; or 1024 / 2048 / 4096 tokens at most. The planner, bible and writer always think up to 4096." }));
                                    if (W("e4_picture_notes")) {
                                        wrap.appendChild(uiHint("Picture notes (optional), one line per connected picture, e.g. '2: the old tailor in a grey kurta'. A hint for a model that sees; the only way to bind pictures for a model that cannot."));
                                        const notes = document.createElement("div");
                                        notes.style.cssText = "margin-left: 28px;";
                                        notes.appendChild(uiTextarea(getW("e4_picture_notes", ""), (v) => setW("e4_picture_notes", v, { rerender: false }), { rows: 3, placeholder: "1: the shopkeeper, an old man in a white kurta\n2: the small shop with the green door" }));
                                        wrap.appendChild(notes);
                                    }
                                }
                            }
                            const replan = document.createElement("button");
                            replan.textContent = "Replan from story";
                            replan.title = "Clear the untouched planned asks (and their rewrites) so the next run plans the story again. Clips you edited or typed stay.";
                            replan.style.cssText = "background: #2a2438; color: #c4b5fd; border: 1px solid #3c3c56; border-radius: 4px; padding: 3px 8px; font-size: 11px; cursor: pointer;";
                            replan.onclick = () => replanFromStory();
                            wrap.appendChild(uiRow("", replan, { indent: true }));
                        }
                    }
                }
                wrap.appendChild(uiRow("greedy decoding", bindToggle("rewrite_greedy")));
                if (!getW("rewrite_greedy", true)) wrap.appendChild(uiRow("temperature", bindNumber("rewrite_temperature"), { indent: true }));
                wrap.appendChild(uiRow("max new tokens", bindNumber("rewrite_max_new_tokens")));
                if (expert()) {
                    wrap.appendChild(uiRow("seed", bindNumber("rewrite_seed")));
                    wrap.appendChild(uiRow("parallel slots", bindNumber("rewrite_parallel"), { hint: "Clip prompts generated at the same time on the server." }));
                }
                const sysWired = Boolean(node.inputs?.find((i) => i.name === "rewrite_system_prompt_in" && i.link != null));
                const sysLabel = document.createElement("div");
                sysLabel.style.cssText = "font-size: 10.5px; color: #8888a4; padding: 2px 4px 0;";
                sysLabel.textContent = sysWired ? "system prompt: fed from the rewrite_system_prompt_in input (text below ignored)" : "system prompt (empty = the Studio builder for Ref2VA, MiniMax's official guide for T2VA; @official = always the official guide)";
                wrap.appendChild(sysLabel);
                const sys = uiTextarea(getW("rewrite_system_prompt", ""), (v) => setW("rewrite_system_prompt", v, { rerender: false }), { rows: 3, placeholder: "Leave empty for the default (Studio builder for Ref2VA), type @official for MiniMax's official guide, or paste your own house style." });
                if (sysWired) sys.style.opacity = "0.5";
                wrap.appendChild(sys);
                wrap.appendChild(uiHint("Rewritten clips show a ✎ badge. They are rewritten again automatically when their raw text, duration, aspect, reference pictures, models or system prompt change; otherwise they are left alone. 'Rewrite again' in a clip's prompt panel forces it."));
                if (writerLinked()) {
                    const reimport = document.createElement("button");
                    reimport.textContent = "Re-import from writer";
                    reimport.title = "Empty the clip list so the next run imports the clips again from the H3 Shot Writer wired into clips_in. Your clips, edits and rewrites in the list are dropped.";
                    reimport.style.cssText = "background: #1f3a2e; color: #86efac; border: 1px solid #2f6b4c; border-radius: 4px; padding: 3px 8px; font-size: 11px; cursor: pointer;";
                    reimport.onclick = () => reimportFromWriter();
                    wrap.appendChild(uiRow("", reimport));
                }
            }

            function section(key, title, summaryFn, buildBody, { collapsible = true } = {}) {
                const wrap = document.createElement("div");
                wrap.style.cssText = "display: flex; flex-direction: column; gap: 5px;";
                const open = collapsible ? Boolean(ui.open[key]) : true;
                wrap.appendChild(uiSectionHeader(title, {
                    summary: open ? "" : summaryFn(),
                    open, collapsible,
                    onToggle: () => { ui.open[key] = !ui.open[key]; renderUI(); },
                }));
                if (open) buildBody(wrap);
                return wrap;
            }

            function buildEngine(wrap) {
                // Turbo LoRAs are trained for about 4 steps; switching engine resets the turbo steps.
                if (presetsMod) {
                    try {
                        const { MASTER_PRESETS, matchPreset } = presetsMod;
                        const matched = matchPreset((n) => W(n)?.value);
                        const sel = document.createElement("select");
                        sel.style.cssText = UI_CONTROL_CSS;
                        const entries = [...(matched ? [] : [{ id: "custom", label: "Custom" }]), ...MASTER_PRESETS];
                        for (const p of entries) {
                            const opt = document.createElement("option");
                            opt.value = p.id;
                            opt.textContent = p.label;
                            if (p.hint) opt.title = p.hint;
                            if ((matched ? matched.id : "custom") === p.id) opt.selected = true;
                            sel.appendChild(opt);
                        }
                        sel.onchange = () => {
                            const p = MASTER_PRESETS.find((x) => x.id === sel.value);
                            if (!p) return;
                            for (const [name, value] of Object.entries(p.values)) setW(name, value, { rerender: false });
                            renderUI();
                        };
                        wrap.appendChild(uiRow("preset", sel, { hint: "Fills the engine widgets below. Shows Custom when the values match no preset. Not saved itself: only the widgets are." }));
                    } catch (e) { console.warn("[H3 Master] preset row skipped", e); }
                }
                const accelW = W("accel_mode");
                const accelSel = uiSelect(comboValues(accelW), accelW?.value, (v) => {
                    setW("accel_mode", v, { rerender: false });
                    setW("pdd_nfe", String(v).toLowerCase().startsWith("turbo") ? "4" : "8", { rerender: false });
                    renderUI();
                });
                wrap.appendChild(uiRow("engine", accelSel, { hint: "Turbo LoRA: any turbo LoRA with its own step count. HyperFlow 8-step: Video Rebirth's 8-step LoRA with two-time conditioning. HyperFlow 8-step \u00b7 dual sampling: HyperFlow with pass 2 as the grid tail over the whole clip in one window." }));
                if (isHyperflow()) {
                    wrap.appendChild(uiRow("hyperflow file", bindSelect("hyperflow_file", { render: shortModelName }),
                        { hint: "Pruned bases (Singularity) need the _pruned build; full bases need the one without. Always the trained 8-step grid with Euler." }));
                    wrap.appendChild(uiRow("curve refit", bindToggle("hyperflow_curve_refit"),
                        { hint: "Pruned base only: restores most of the two-time (t, r) conditioning. Off = LoRA-only." }));
                    if (expert()) wrap.appendChild(uiRow("strength", bindNumber("hyperflow_strength"), { indent: true }));
                    wrap.appendChild(uiHint("Pass 2 without a refine LoRA is cut from HyperFlow's own grid; a refine LoRA forces replace mode (no HyperFlow on that tail)."));
                    const dualMode = isDualMode(getW("accel_mode", ""));
                    if (W(DUAL_WIDGET)) {
                        if (!dualMode) wrap.appendChild(uiRow("dual sampling", bindToggle(DUAL_WIDGET),
                            { hint: "Pass 2 = the grid tail from the split step, generated at the pass-2 size; pass 1 only sets the layout. No refine LoRA, refine denoise ignored." }));
                        if (dualMode || getW(DUAL_WIDGET, false)) {
                            wrap.appendChild(uiRow("pass 1 early stop", bindToggle("pass1_early_stop"),
                                { indent: true, hint: "Experimental: pass 1 runs only the first split-step steps. The 3D upscaler was trained on clean latents." }));
                        }
                    }
                    if (W(PASS1_STEPS_WIDGET) && !(dualMode || getW(DUAL_WIDGET, false))) {
                        wrap.appendChild(uiRow("pass 1 steps", bindNumber(PASS1_STEPS_WIDGET),
                            { hint: "0 = all 8 HyperFlow steps. 1-7: stop pass 1 early, upscale its clean estimate, then the normal refine (same LoRA, windows, denoise). Saves the skipped steps." }));
                    }
                } else if (isTurbo()) {
                    wrap.appendChild(uiRow("turbo lora", bindSelect("turbo_lora", { render: shortModelName })));
                    wrap.appendChild(uiRow("steps", bindSelect("pdd_nfe")));
                    if (expert()) {
                        wrap.appendChild(uiRow("lora strength", bindNumber("turbo_lora_strength"), { indent: true }));
                        wrap.appendChild(uiRow("sampler", bindSelect("turbo_sampler"), { indent: true }));
                        wrap.appendChild(uiRow("scheduler", bindSelect("turbo_scheduler"), { indent: true }));
                    } else {
                        wrap.appendChild(uiHint(`sampler ${getW("turbo_sampler", "")} / ${getW("turbo_scheduler", "")}, strength ${Number(getW("turbo_lora_strength", 1)).toFixed(2)} — switch to Expert to change`));
                    }
                }
                // Pass 2: with dual sampling one row edits the split step and the refine LoRA is not used; otherwise the refine LoRA + schedule.
                const dualActive = isHyperflow() && Boolean(W(DUAL_WIDGET)) && (isDualMode(getW("accel_mode", "")) || Boolean(getW(DUAL_WIDGET, false)));
                const p2b = pass2RowBinding(dualActive, getW("dual_split_step", 2));
                if (dualActive) {
                    wrap.appendChild(uiRow("pass 2 steps", bindNumber(p2b.widget), { hint: p2b.hint }));
                    wrap.appendChild(uiHint("Refine lora, strength and mode are ignored in dual sampling."));
                } else if (isOpenH3()) {
                    wrap.appendChild(uiRow("refiner", bindSelect("pass2_refiner")));
                    wrap.appendChild(uiHint("Open-H3 refiner LoRA alone, 3 steps from sigma 0.909 to the refine resolution. Faces are found in the draft and start at 0.50, so they keep their identity; the pass-1 audio is kept. Refine lora, steps and denoise are not used."));
                } else {
                    const p2lora = String(getW("pass2_lora", "none"));
                    if (W("pass2_refiner")) wrap.appendChild(uiRow("refiner", bindSelect("pass2_refiner")));
                    wrap.appendChild(uiRow("refine lora", bindSelect("pass2_lora", { render: shortModelName }),
                        { hint: "LoRA applied only on the pass-2 refine tail. none = same model as pass 1." }));
                    if (p2lora !== "none") {
                        wrap.appendChild(uiRow("refine lora strength", bindNumber("pass2_lora_strength"), { indent: true }));
                        wrap.appendChild(uiRow("mode", bindSelect("pass2_lora_mode", { render: (v) => String(v).replace(" engine LoRA", "") }),
                            { indent: true, hint: "stack: on top of the engine LoRA. replace: base model + this LoRA only (step-distilled LoRAs)." }));
                        wrap.appendChild(uiRow("pass 2 steps", bindNumber(p2b.widget), { indent: true, hint: p2b.hint }));
                    }
                }
                // Semantic bridge (BUNNY H3 Conditioning Bridge): a small residual MLP on the
                // text conditioning of both passes. Only shown on node builds that have the widget.
                if (W("semantic_bridge")) {
                    const bridge = String(getW("semantic_bridge", "none"));
                    wrap.appendChild(uiRow("semantic bridge", bindSelect("semantic_bridge", { render: shortModelName }),
                        { hint: "BUNNY bridge on the conditioning of both passes: who does what to whom, object ownership, state continuity. none = off." }));
                    if (bridge !== "none") {
                        wrap.appendChild(uiRow("bridge alpha", bindNumber("semantic_bridge_alpha"),
                            { indent: true, hint: "Residual strength. 0.10-0.15 recommended; higher is not better." }));
                        if (expert()) wrap.appendChild(uiRow("magnitude match", bindSelect("semantic_bridge_match"), { indent: true }));
                    }
                }
            }

            function buildQuality(wrap) {
                const { preset, orientation } = currentPreset();
                const presetSel = uiSelect(
                    [...Object.keys(QUALITY_PRESETS), "custom"], preset,
                    (v) => {
                        if (v === "custom") { ui.qualityCustom = true; renderUI(); }
                        else applyQualityPreset(v, orientation);
                    },
                    { render: (v) => v === "custom" ? "Custom" : QUALITY_PRESETS[v].label },
                );
                wrap.appendChild(uiRow("preset", presetSel));
                const orient = uiSegmented(ORIENTATIONS, orientation, (o) => applyQualityPreset(preset === "custom" ? "standard" : preset, o));
                wrap.appendChild(uiRow("orientation", orient));
                const custom = preset === "custom" || expert();
                if (custom) {
                    wrap.appendChild(uiRow("draft resolution", bindSelect("pass1_resolution"), { indent: true }));
                    wrap.appendChild(uiRow("refine resolution", bindSelect("pass2_resolution"), { indent: true }));
                    wrap.appendChild(uiRow("refine denoise", bindNumber("pass2_denoise"), { indent: true }));
                } else {
                    wrap.appendChild(uiHint(qualitySummary().split(" · ").slice(2).join(" · ")));
                }
                // The latent upscaler is a per-render choice independent of the preset, so it
                // stays visible in Simple mode too (models/latent_upscale_models).
                wrap.appendChild(uiRow("upscaler", bindSelect("upscaler_model", { render: shortModelName }),
                    { hint: "3D latent upscaler between the draft and refine passes." }));
                // Final decode changes the picture (2x output), so it lives with the other quality
                // choices, not under Performance. Hidden unless a packed 2x VAE is installed.
                if (comboValues(W("final_decode")).length > 1) {
                    wrap.appendChild(uiRow("final decode", bindSelect("final_decode", { render: (v) => v === "standard" ? v : shortModelName(v) }),
                        { hint: "standard = decode with the connected VAE. A 2x VAE decodes the final video at 2x resolution, ~+23 s per 15 s clip; needs ComfyUI-MiniMaxH3_LatentUpscaler. Guide frames and previews stay at the normal resolution." }));
                }
            }

            function buildContinuity(wrap) {
                wrap.appendChild(uiRow("motion context frames", bindSelect("context_length"), { hint: "Video frames from the previous clip fed into the next one." }));
                wrap.appendChild(uiRow("audio context frames", bindNumber("audio_context_length")));
                wrap.appendChild(uiRow("identity continuity", bindToggle("identity_continuity"), { hint: "Use an empty picture slot for the previous clip's last frame." }));
            }

            function buildPerformance(wrap) {
                wrap.appendChild(uiRow("attention", bindSelect("attention_backend", { render: (v) => String(v).replace(" attention", "") })));
                wrap.appendChild(uiRow("sparse attention (SLA)", bindToggle("sla_enabled")));
                if (getW("sla_enabled", false)) {
                    wrap.appendChild(uiRow("sparsity", bindNumber("sla_sparsity"), { indent: true }));
                    if (W("pass2_sla_sparsity")) wrap.appendChild(uiRow("pass 2 SLA sparsity (0 = same)", bindNumber("pass2_sla_sparsity"), { indent: true,
                        hint: "Sparsity for the refine pass only (sla / vsa methods). 0.95 keeps 5% of key blocks in pass 2 while pass 1 keeps the sparsity above." }));
                    wrap.appendChild(uiRow("method", bindSelect("sparse_method"), { indent: true }));
                    if (getW("sparse_method", "sla") === "sol-attn") wrap.appendChild(uiRow("tau", bindNumber("sparse_tau"), { indent: true, hint: "HyperFlow's validated recipe is tau 1.0." }));
                    const recipe = isHyperflow() && getW("sparse_method", "sla") === "sol-attn";
                    const autoNote = recipe ? "auto = HyperFlow sol-attn recipe" : "auto = default";
                    wrap.appendChild(uiRow("sparse start %", bindNumber("sparse_start_percent"), { indent: true,
                        hint: `-1 = auto (${autoNote}: ${recipe ? "0.16" : "0.20"}). Steps before this fraction of the schedule run dense.` }));
                    wrap.appendChild(uiRow("dense blocks", uiText(getW("sparse_dense_blocks", "auto"), (v) => setW("sparse_dense_blocks", v)), { indent: true,
                        hint: `Transformer blocks that always run dense, e.g. 0,1. auto (${autoNote}: ${recipe ? "0,1" : "none"}); empty = none.` }));
                    wrap.appendChild(uiRow("sink", bindSelect("sparse_sink"), { indent: true,
                        hint: `Keep text/audio/reference rows exact. auto (${autoNote}: ${recipe ? "off" : "exact_kv_and_rows"}).` }));
                }
                const chunkOn = Number(getW("pass2_chunk_frames", 0)) > 0;
                wrap.appendChild(uiRow("chunked refine pass", uiToggle(chunkOn, (on) => {
                    if (on) setW("pass2_chunk_frames", Number(node.properties.h3_last_chunk_frames) || 124);
                    else { node.properties.h3_last_chunk_frames = getW("pass2_chunk_frames", 124); setW("pass2_chunk_frames", 0); }
                }), { hint: "Refine long clips in temporal windows instead of one attention pass. By default only used above 720p." }));
                if (chunkOn) {
                    wrap.appendChild(uiRow("chunk frames", bindNumber("pass2_chunk_frames"), { indent: true }));
                    wrap.appendChild(uiRow("overlap", bindNumber("pass2_chunk_overlap"), { indent: true }));
                    if (W("pass2_chunk_when")) {
                        wrap.appendChild(uiRow("chunk when", bindSelect("pass2_chunk_when"), { indent: true,
                            hint: "above 720p = window only when the refine short side is over 800 px (1080p-class); 720p refines the whole clip. always = window any clip longer than chunk frames." }));
                    }
                }
                // Audio-only refine is not offered in the panel (founder, 2026-10-01: "we dont need the
                // audio refine at all"). Its inputs stay on the node, default 0 = off, because removing
                // them would shift every later saved widget value (final_decode, master_ui).
                wrap.appendChild(uiRow("offload upscaler after use", bindToggle("smart_offload")));
                wrap.appendChild(uiRow("background decode (experimental)", bindSelect("async_decode")));
            }

            function fitNodeToContent() {
                requestAnimationFrame(() => {
                    const need = container.scrollHeight - container.clientHeight;
                    if (need > 2) node.setSize([node.size[0], node.size[1] + need + 4]);
                });
            }

            // ---- Prompt side panel: one per node, lives on document.body ----
            const promptPanel = { el: null, clipId: null, raf: 0, textarea: null, layer: null, title: null, meta: null, view: "auto", viewClipId: null };
            // Rewrites in flight, by clip id: { phase, text } -- shown live in the panel and on the card.
            const liveRewrite = new Map();

            function rawTextOf(clip) {
                return typeof clip.prompt_raw === "string" ? clip.prompt_raw : (clip.prompt || "");
            }
            function rewrittenTextOf(clip) {
                if (typeof clip.rewrite_text === "string") return clip.rewrite_text;
                return clip.prompt_rewritten ? (clip.prompt || "") : "";
            }
            // Drop a clip's rewrite (text, flag, meta, live stream) so it is pending
            // from its raw text. With continuity on, later clips were written from this
            // clip's raw ask, so theirs go too. Returns true when anything was dropped.
            function wipeRewrite(clip) {
                const had = clip.prompt_rewritten || typeof clip.rewrite_text === "string" || clip.rewrite_meta || liveRewrite.has(clip.id);
                if (!had) return false;
                if (clip.prompt_rewritten && typeof clip.prompt_raw === "string") clip.prompt = clip.prompt_raw;
                clip.prompt_rewritten = false;
                delete clip.rewrite_text;
                delete clip.rewrite_meta;
                liveRewrite.delete(clip.id);
                clip.validated = false;
                return true;
            }
            // The one explicit way to plan again: drop the plan's untouched asks (they were never typed by hand) and
            // every planned flag, so the next run's planner sees empty slots. Edited planned clips become hand clips.
            function replanFromStory() {
                const untouched = clipsState.filter((c) => c.planned === true);
                if (!clipsState.some((c) => "planned" in c)) { alert("Nothing was planned yet; the next run plans the story."); return; }
                if (!confirm(`Replan from the story? ${untouched.length} untouched planned clip(s) are emptied (their rewrites and renders are dropped); clips you edited or typed stay. Then queue the workflow to plan again.`)) return;
                for (const clip of clipsState) {
                    if (clip.planned === true) {
                        clip.prompt = "";
                        delete clip.prompt_raw;
                        clip.prompt_rewritten = false;
                        delete clip.rewrite_text;
                        delete clip.rewrite_meta;
                        liveRewrite.delete(clip.id);
                        clip.validated = false;
                    }
                    delete clip.planned;
                    if (clip.prompt === "") delete clip.plan_end_state;
                }
                saveState();
                renderUI();
            }
            function writerLinked() {
                return Boolean(node.inputs?.find((i) => i.name === "clips_in" && i.link != null));
            }
            // The next run imports the clips again when the list holds no prompt (the backend asks the Shot Writer only then).
            function reimportFromWriter() {
                if (!confirm("Empty the clip list so the next run imports the clips again from the writer? Every clip in the list, with its edits, rewrites and renders, is dropped.")) return;
                const fresh = { id: 0, title: "Clip 1", prompt: "", duration: DUR_DEFAULT, seed: Math.floor(Math.random() * 1000000000), seed_mode: "randomize", validated: false, loras: [] };
                liveRewrite.clear();
                clipsState.splice(0, clipsState.length, fresh);
                saveState();
                renderUI();
            }
            // Every clip back to its raw prompt; the raw prompts themselves are left as they are.
            function clearAllRewrites() {
                const rewritten = clipsState.filter((c) => c.prompt_rewritten || typeof c.rewrite_text === "string" || c.rewrite_meta || liveRewrite.has(c.id));
                if (!rewritten.length) { alert("No clip has a rewritten prompt."); return; }
                if (!confirm(`Clear the rewritten prompts of ${rewritten.length} clip(s)? Their raw prompts stay; the next run rewrites them again.`)) return;
                for (const clip of clipsState) wipeRewrite(clip);
                saveState();
                renderUI();
            }
            function wipeRewrites(clip) {
                let dropped = wipeRewrite(clip);
                const continuity = node.widgets?.find((w) => w.name === "rewrite_previous_clips")?.value;
                if (continuity && continuity !== "off") {
                    const { index } = findClip(clip.id);
                    for (const later of clipsState.slice(index + 1)) dropped = wipeRewrite(later) || dropped;
                }
                return dropped;
            }
            function panelView(clip) {
                if (promptPanel.viewClipId === clip.id && (promptPanel.view === "raw" || promptPanel.view === "rewritten")) return promptPanel.view;
                return (liveRewrite.has(clip.id) || clip.prompt_rewritten || rewrittenTextOf(clip)) ? "rewritten" : "raw";
            }

            function markCardSelected(card, on) {
                card.style.outline = on ? "2px solid #9d8cff" : "none";
                card.style.outlineOffset = on ? "-1px" : "0";
            }

            function findClip(clipId) {
                const index = clipsState.findIndex((c) => c.id === clipId);
                return { index, clip: index >= 0 ? clipsState[index] : null };
            }

            function closePromptPanel() {
                if (promptPanel.raf) cancelAnimationFrame(promptPanel.raf);
                promptPanel.raf = 0;
                promptPanel.el?.remove();
                promptPanel.el = null;
                promptPanel.textarea = null;
                promptPanel.clipId = null;
                container.querySelectorAll(".minimax-clip-card").forEach((c) => markCardSelected(c, false));
            }

            function buildPromptPanel() {
                ensureHighlightStyles();
                const el = document.createElement("div");
                el.className = "minimax-prompt-panel";
                el.style.cssText = `
                    position: fixed; z-index: 9000;
                    display: flex; flex-direction: column; gap: 8px;
                    background: #14141c; border: 1px solid #383852; border-radius: 10px;
                    padding: 10px 12px; box-shadow: 0 14px 44px rgba(0,0,0,0.55);
                    color: #e2e2ec; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: 12px;
                    user-select: text;
                `;
                el.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center; gap: 8px;">
                        <div style="display: flex; align-items: center; gap: 6px;">
                            <button class="pp-prev" title="Previous clip" style="background: #252536; border: 1px solid #3f3f58; color: #c7c7e0; border-radius: 3px; font-size: 11px; padding: 1px 7px; cursor: pointer;">&#9664;</button>
                            <span class="pp-title" style="font-size: 13px; font-weight: 700; color: #9d8cff;"></span>
                            <button class="pp-next" title="Next clip" style="background: #252536; border: 1px solid #3f3f58; color: #c7c7e0; border-radius: 3px; font-size: 11px; padding: 1px 7px; cursor: pointer;">&#9654;</button>
                        </div>
                        <div style="display: flex; align-items: center; gap: 10px;">
                            <span class="pp-meta" style="color: #6f6f8a; font-size: 11px; font-variant-numeric: tabular-nums;"></span>
                            <span class="pp-view" style="display: inline-flex; border: 1px solid #3f3f58; border-radius: 4px; overflow: hidden;">
                                <button data-view="raw" style="background: #252536; border: none; color: #c7c7e0; font-size: 10.5px; padding: 2px 8px; cursor: pointer;">raw</button>
                                <button data-view="rewritten" style="background: #252536; border: none; border-left: 1px solid #3f3f58; color: #c7c7e0; font-size: 10.5px; padding: 2px 8px; cursor: pointer;">rewritten</button>
                            </span>
                            <button class="pp-raw" title="Mark this clip pending: the rewriter writes it again from the raw text on the next run (the current rewrite stays visible until then)" style="display: none; background: #3a2e14; border: 1px solid #6b5416; color: #fcd34d; border-radius: 3px; font-size: 10.5px; padding: 1px 7px; cursor: pointer;">Rewrite again</button>
                            <button class="pp-close" title="Close (Esc)" style="background: transparent; border: none; color: #9c9cb2; font-size: 15px; cursor: pointer; padding: 0 2px;">&#10005;</button>
                        </div>
                    </div>
                    <div class="pp-editor" style="position: relative; flex: 1; min-height: 0; border: 1px solid #2e2e42; border-radius: 6px; background: #0f0f16; overflow: hidden;">
                        <pre class="pp-layer" aria-hidden="true" style="position: absolute; inset: 0; margin: 0; padding: 12px 14px; overflow: auto; pointer-events: none; white-space: pre-wrap; word-break: break-word; color: #d9d9e6; font: 13px/1.6 'Cascadia Mono', Consolas, 'JetBrains Mono', ui-monospace, monospace; tab-size: 2;"></pre>
                        <textarea class="pp-text" spellcheck="false" style="position: absolute; inset: 0; width: 100%; height: 100%; box-sizing: border-box; margin: 0; padding: 12px 14px; border: none; outline: none; resize: none; background: transparent; color: transparent; caret-color: #ffffff; overflow: auto; white-space: pre-wrap; word-break: break-word; font: 13px/1.6 'Cascadia Mono', Consolas, 'JetBrains Mono', ui-monospace, monospace; tab-size: 2; user-select: text; -webkit-user-select: text;"></textarea>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: center; color: #6f6f8a; font-size: 10.5px;">
                        <span class="pp-foot">Saves as you type · Esc closes · Tab inserts two spaces</span>
                        <span style="display: flex; gap: 8px;">
                            <span><span class="h3-section">## section</span></span>
                            <span><span class="h3-key">key</span><span class="h3-colon">:</span></span>
                            <span class="h3-ref">&lt;Picture 1&gt;</span>
                            <span class="h3-tag">[Shot 1]</span>
                            <span class="h3-time">00:04.500</span>
                        </span>
                    </div>
                `;
                document.body.appendChild(el);

                const textarea = el.querySelector(".pp-text");
                const layer = el.querySelector(".pp-layer");
                const paint = () => { layer.innerHTML = highlightH3(textarea.value); };
                const syncScroll = () => { layer.scrollTop = textarea.scrollTop; layer.scrollLeft = textarea.scrollLeft; };

                let saveTimer = 0;
                textarea.oninput = () => {
                    paint();
                    const { clip } = findClip(promptPanel.clipId);
                    if (!clip || textarea.readOnly) return;
                    const value = textarea.value;
                    if (panelView(clip) === "raw") {
                        // Editing the original: the clip is pending again and renders from
                        // this text until the rewriter has replaced it. The old rewrite is
                        // wiped so it can never be rendered for the new text.
                        clip.prompt_raw = value;
                        clip.prompt = value;
                        if (clip.planned === true) clip.planned = false;
                        if (wipeRewrites(clip)) renderUI();
                    } else {
                        // Editing the rewritten text: used as is, not rewritten again.
                        clip.rewrite_text = value;
                        clip.prompt = value;
                    }
                    clip.validated = false;
                    refreshPanelChrome(clip);
                    const card = [...container.querySelectorAll(".minimax-clip-card")].find((c) => c._clipId === clip.id);
                    card?._paintPreview?.();
                    clearTimeout(saveTimer);
                    saveTimer = setTimeout(saveState, 150);
                };
                el.querySelectorAll(".pp-view button").forEach((b) => {
                    b.onclick = () => {
                        const { clip } = findClip(promptPanel.clipId);
                        if (!clip) return;
                        promptPanel.view = b.dataset.view;
                        promptPanel.viewClipId = clip.id;
                        openPromptPanel(clip.id, { focus: false });
                    };
                });
                textarea.onscroll = syncScroll;
                textarea.addEventListener("keydown", (e) => {
                    if (e.key === "Escape") { e.preventDefault(); closePromptPanel(); return; }
                    if (e.key === "Tab") {
                        e.preventDefault();
                        const { selectionStart: a, selectionEnd: b } = textarea;
                        textarea.setRangeText("  ", a, b, "end");
                        textarea.oninput();
                    }
                    e.stopPropagation();   // no graph shortcuts while typing
                });
                el.addEventListener("wheel", (e) => e.stopPropagation(), { passive: true });
                el.addEventListener("pointerdown", (e) => e.stopPropagation());
                el.querySelector(".pp-close").onclick = closePromptPanel;
                el.querySelector(".pp-raw").onclick = () => {
                    const { clip } = findClip(promptPanel.clipId);
                    if (!clip) return;
                    if (typeof clip.prompt_raw === "string" && clip.prompt_raw.trim()) clip.prompt = clip.prompt_raw;
                    clip.prompt_rewritten = false;
                    if (clip.rewrite_meta) clip.rewrite_meta = Object.assign({}, clip.rewrite_meta, { stale: true });
                    clip.validated = false;
                    saveState();
                    renderUI();
                    promptPanel.view = "raw";
                    promptPanel.viewClipId = clip.id;
                    openPromptPanel(clip.id, { focus: true });
                };
                el.querySelector(".pp-prev").onclick = () => stepPromptPanel(-1);
                el.querySelector(".pp-next").onclick = () => stepPromptPanel(1);

                promptPanel.el = el;
                promptPanel.textarea = textarea;
                promptPanel.layer = layer;
                promptPanel.title = el.querySelector(".pp-title");
                promptPanel.meta = el.querySelector(".pp-meta");
                promptPanel.paint = paint;

                const follow = () => {
                    if (!promptPanel.el) return;
                    if (!node.graph) { closePromptPanel(); return; }   // node was deleted
                    placePanelNextToNode(el, node);
                    promptPanel.raf = requestAnimationFrame(follow);
                };
                follow();
            }

            function stepPromptPanel(delta) {
                const { index } = findClip(promptPanel.clipId);
                if (index < 0) return;
                const next = clipsState[(index + delta + clipsState.length) % clipsState.length];
                openPromptPanel(next.id);
            }

            // Header, toggle, footer and read-only state for the clip in the panel.
            function refreshPanelChrome(clip) {
                if (!promptPanel.el) return;
                const view = panelView(clip);
                const live = liveRewrite.get(clip.id);
                const stale = !clip.prompt_rewritten && Boolean(rewrittenTextOf(clip));
                promptPanel.el.querySelectorAll(".pp-view button").forEach((b) => {
                    const on = b.dataset.view === view;
                    b.style.background = on ? "#6355d8" : "#252536";
                    b.style.color = on ? "#ffffff" : "#c7c7e0";
                    if (b.dataset.view === "rewritten") {
                        b.textContent = live ? (live.phase === "thinking" ? "rewritten · thinking…" : "rewritten · writing…")
                            : stale ? "rewritten (stale)" : (clip.prompt_rewritten ? "rewritten ✎" : "rewritten");
                    }
                });
                const rawBtn = promptPanel.el.querySelector(".pp-raw");
                if (rawBtn) rawBtn.style.display = view === "rewritten" && clip.prompt_rewritten && !live ? "" : "none";
                const foot = promptPanel.el.querySelector(".pp-foot");
                let meta;
                if (view === "raw") {
                    meta = (clip.prompt_rewritten ? "original · rendered from the rewrite" : "original · pending rewrite") + " · " + describePrompt(rawTextOf(clip));
                    if (foot) foot.textContent = clip.prompt_rewritten
                        ? "Editing the original marks the clip pending again · Esc closes"
                        : "Saves as you type · rewritten at the start of the next run · Esc closes";
                } else if (live) {
                    meta = `${live.phase === "thinking" ? "thinking" : "writing"} · ${(live.text || "").length} chars`;
                    if (foot) foot.textContent = "Live from the rewriter · read-only until it finishes";
                } else if (stale) {
                    meta = "stale · original edited since · " + describePrompt(rewrittenTextOf(clip));
                    if (foot) foot.textContent = "This rewrite is stale: the original changed, the clip renders from the original until the next run rewrites it";
                } else if (clip.prompt_rewritten) {
                    const m = clip.rewrite_meta || {};
                    meta = `✎ ${m.model ? modelLabel(m.model).split(".")[0] : "rewritten"}${m.thinking ? " · thinking" : ""}${m.seconds ? ` · ${m.seconds}s` : ""} · ${describePrompt(clip.prompt || "")}`;
                    if (foot) foot.textContent = "Edits here are used as is (not rewritten again) · 'Rewrite again' marks the clip pending · Esc closes";
                } else {
                    meta = "not rewritten yet";
                    if (foot) foot.textContent = "The rewriter fills this at the start of the next run (Prompt Rewriter section: mode ≠ off)";
                }
                promptPanel.meta.textContent = meta;
            }

            function openPromptPanel(clipId, { focus = true } = {}) {
                const { index, clip } = findClip(clipId);
                if (!clip) return;
                if (!promptPanel.el) buildPromptPanel();
                if (promptPanel.clipId !== clipId) { promptPanel.view = "auto"; promptPanel.viewClipId = null; }
                promptPanel.clipId = clipId;
                promptPanel.title.textContent = `Clip ${index + 1}${clip.title && clip.title !== `Clip ${index + 1}` ? ` · ${clip.title}` : ""}`;
                const view = panelView(clip);
                const live = liveRewrite.get(clip.id);
                let text;
                if (view === "raw") text = rawTextOf(clip);
                else if (live) text = live.text || (live.phase === "thinking" ? "(thinking…)" : "(writing…)");
                else text = rewrittenTextOf(clip);
                const stale = view === "rewritten" && !clip.prompt_rewritten && Boolean(rewrittenTextOf(clip));
                promptPanel.textarea.readOnly = view === "rewritten" && (Boolean(live) || stale || !clip.prompt_rewritten);
                promptPanel.textarea.style.caretColor = promptPanel.textarea.readOnly ? "transparent" : "#ffffff";
                promptPanel.layer.style.opacity = live && live.phase === "thinking" ? "0.6" : (stale ? "0.7" : "1");
                promptPanel.textarea.value = text;
                if (!text && view === "rewritten") {
                    promptPanel.layer.innerHTML = `<span style="color: #55556e;">Not rewritten yet. With the Prompt Rewriter on, this fills in live at the start of the next run.</span>`;
                } else {
                    promptPanel.paint();
                }
                refreshPanelChrome(clip);
                promptPanel.textarea.scrollTop = 0;
                promptPanel.layer.scrollTop = 0;
                container.querySelectorAll(".minimax-clip-card").forEach((c) => markCardSelected(c, c._clipId === clipId));
                if (focus && !promptPanel.textarea.readOnly) promptPanel.textarea.focus({ preventScroll: true });
            }

            const originalOnRemoved = node.onRemoved;
            node.onRemoved = function () {
                closePromptPanel();
                return originalOnRemoved?.apply(this, arguments);
            };

            // clip_prompt_N inputs let another node supply a clip's prompt.
            function externalPromptWired(index) {
                const input = node.inputs?.find((i) => i.name === `clip_prompt_${index + 1}`);
                return Boolean(input && input.link != null);
            }
            const originalOnConnectionsChange = node.onConnectionsChange;
            node.onConnectionsChange = function (type, slotIndex, connected, linkInfo, ioSlot) {
                const r = originalOnConnectionsChange?.apply(this, arguments);
                const changed = String(ioSlot?.name || "");
                if (changed.startsWith("clip_prompt_") || changed.startsWith("ref_video") || changed.startsWith("ref_audio") || changed === "rewrite_system_prompt_in") renderUI();
                return r;
            };

            function renderUI() {
                container.innerHTML = "";

                // 1. Master Header
                const header = document.createElement("div");
                header.style.cssText = `
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    border-bottom: 1px solid #2e2e42;
                    padding-bottom: 10px;
                `;

                const totalSec = clipsState.reduce((acc, c) => acc + Number(c.duration || DUR_DEFAULT), 0);
                const totalFrames = clipsState.reduce((acc, c) => acc + alignH3Frames(c.duration || DUR_DEFAULT), 0);

                header.innerHTML = `
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <span style="font-size: 15px; font-weight: 700; color: #9d8cff; letter-spacing: -0.2px;">🎬 MiniMax H3 Master</span>
                        <span style="background: #252538; color: #a5a5c5; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 500; border: 1px solid #383852;">${isDualMode(getW("accel_mode", "")) ? "HyperFlow · dual sampling" : `${isHyperflow() ? "HyperFlow" : "Turbo"} · ${isHyperflow() ? "8" : getW("pdd_nfe", "8")} steps · 2-pass`}</span>
                        <button id="mode-toggle" title="Simple shows the eight decisions that matter; Expert shows every field." style="background: ${expert() ? "#4a3c22" : "#2b3a55"}; color: ${expert() ? "#e6c28f" : "#8fb0e6"}; border: 1px solid ${expert() ? "#6b5530" : "#3c5480"}; padding: 2px 9px; border-radius: 10px; font-size: 10.5px; font-weight: 600; cursor: pointer;">${expert() ? "Expert" : "Simple"}</button>
                    </div>
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="color: #9292ab; font-size: 11px; font-weight: 500;">
                            ${clipsState.length} Clips (${totalSec.toFixed(1)}s / ${totalFrames} frames)
                        </span>
                    </div>
                `;
                container.appendChild(header);
                header.querySelector("#mode-toggle").onclick = () => {
                    ui.mode = expert() ? "simple" : "expert";
                    renderUI();
                };
                container.appendChild(projectControls.toolbar());

                // 2. Engine + Quality
                container.appendChild(section("engine", "Engine", engineSummary, buildEngine));
                container.appendChild(section("quality", "Quality", qualitySummary, buildQuality));

                const addClip = () => {
                    const nextId = nextClipId(clipsState);
                    clipsState.push({
                        id: nextId,
                        title: `Clip ${nextId + 1}`,
                        prompt: "",
                        duration: DUR_DEFAULT,
                        beyond: false,
                        seed: Math.floor(Math.random() * 1000000000),
                        seed_mode: "randomize",
                        validated: false,
                        loras: []
                    });
                    saveState();
                    renderUI();
                };

                // 3. Continuity (expert only) + Performance sit above the clips, so the
                // clip cards are the last thing before the run bar.
                if (expert()) container.appendChild(section("continuity", "Continuity", continuitySummary, buildContinuity));
                container.appendChild(section("performance", "Performance", performanceSummary, buildPerformance));
                container.appendChild(section("rewriter", "Prompt Rewriter", rewriterSummary, buildRewriter));

                // 4. Clips: header carries the Add Clip button so it sits next to the cards.
                {
                    const totalSecClips = clipsState.reduce((acc, c) => acc + Number(c.duration || DUR_DEFAULT), 0);
                    const addBtn = document.createElement("button");
                    addBtn.textContent = "+ Add Clip";
                    addBtn.style.cssText = "background: #6355d8; color: #ffffff; border: none; padding: 3px 10px; border-radius: 5px; cursor: pointer; font-weight: 600; font-size: 11px; box-shadow: 0 2px 4px rgba(99,85,216,0.3); white-space: nowrap;";
                    addBtn.onclick = (e) => { e.stopPropagation(); addClip(); };
                    // Remove every clip in one go (long or collapsed sets); one empty clip is left.
                    const clearBtn = document.createElement("button");
                    clearBtn.textContent = "Remove all";
                    clearBtn.title = "Delete every clip and start again from one empty clip";
                    clearBtn.style.cssText = "background: transparent; color: #ef4444; border: 1px solid #5a2a2a; padding: 3px 10px; border-radius: 5px; cursor: pointer; font-weight: 600; font-size: 11px; white-space: nowrap;";
                    clearBtn.onclick = (e) => {
                        e.stopPropagation();
                        if (!confirm(`Remove all ${clipsState.length} clip${clipsState.length === 1 ? "" : "s"}? Their prompts are lost (rendered clips stay in the disk cache).`)) return;
                        closePromptPanel();
                        liveRewrite.clear();
                        expandedValidated.clear();
                        cacheStates = [];
                        cacheVideos = [];
                        clipsState.length = 0;
                        addClip();
                    };
                    // Bulk validation, so a long chain is not toggled one card at a time.
                    const bulkStyle = "background: transparent; border: 1px solid #2e6a45; color: #4ade80; padding: 3px 10px; border-radius: 5px; cursor: pointer; font-weight: 600; font-size: 11px; white-space: nowrap;";
                    const validateAllBtn = document.createElement("button");
                    validateAllBtn.textContent = "Validate all";
                    validateAllBtn.title = "Mark every clip that is on disk as validated (clips not rendered yet are left alone)";
                    validateAllBtn.style.cssText = bulkStyle;
                    validateAllBtn.onclick = (e) => {
                        e.stopPropagation();
                        clipsState.forEach((clip, i) => {
                            const state = cacheStates[i];
                            if (!cacheStates.length || (state && state !== "none")) clip.validated = true;
                        });
                        expandedValidated.clear();
                        saveState();
                        renderUI();
                    };
                    // Char refs, turbo LoRA and engine settings are not part of a clip's cache
                    // fingerprint, so un-ticking alone would hand back the old renders. Clear the
                    // chain on disk too, so the next run re-renders from clip 1 with current settings.
                    const invalidateAllBtn = document.createElement("button");
                    invalidateAllBtn.textContent = "Invalidate all";
                    invalidateAllBtn.title = "Un-validate every clip and clear this chain's disk cache, so the next run re-renders the whole chain with the current references and settings";
                    invalidateAllBtn.style.cssText = bulkStyle.replace("#2e6a45", "#6a5a2e").replace("#4ade80", "#fbbf24");
                    invalidateAllBtn.onclick = async (e) => {
                        e.stopPropagation();
                        if (!confirm(`Invalidate all ${clipsState.length} clip${clipsState.length === 1 ? "" : "s"}? This deletes the rendered chain from the disk cache; the next run re-renders every clip. Prompts, seeds and settings are kept.`)) return;
                        try {
                            const response = await api.fetchApi("/minimax_master/clear_cache", {
                                method: "POST", headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ clips: [JSON.stringify(clipsState)], final_ids: [] }),
                            });
                            if (!response.ok) {
                                const detail = await response.text().catch(() => "");
                                throw new Error(response.status === 409 ? "a job is queued or running - wait for it to finish" : `${response.status} ${detail}`);
                            }
                        } catch (error) {
                            alert(`Could not clear the chain cache: ${error.message}. Nothing was invalidated.`);
                            return;
                        }
                        clipsState.forEach((clip) => { clip.validated = false; });
                        expandedValidated.clear();
                        cacheStates = [];
                        cacheVideos = [];
                        saveState();
                        renderUI();
                    };
                    const actions = document.createElement("span");
                    actions.style.cssText = "display: inline-flex; gap: 6px; align-items: center; flex-wrap: wrap;";
                    // Every clip back to its raw prompt, so the next run rewrites them all (says so when no clip has a rewrite).
                    const clearRewritesBtn = document.createElement("button");
                    clearRewritesBtn.textContent = "Clear rewrites";
                    clearRewritesBtn.title = "Drop the rewritten prompt of every clip and put its raw prompt back, so the next run rewrites them all. Raw prompts are not touched.";
                    clearRewritesBtn.style.cssText = bulkStyle.replace("#2e6a45", "#6b5416").replace("#4ade80", "#fcd34d");
                    clearRewritesBtn.onclick = (e) => { e.stopPropagation(); clearAllRewrites(); };
                    actions.append(clearRewritesBtn, validateAllBtn, invalidateAllBtn, clearBtn, addBtn);
                    container.appendChild(uiSectionHeader("Clips", { summary: `${clipsState.length} clip${clipsState.length === 1 ? "" : "s"} · ${totalSecClips.toFixed(0)} s`, collapsible: false, actions }));
                }
                container.appendChild(projectControls.references());
                {
                    // Video / audio references arrive on sockets; say which are wired.
                    const wired = (prefix, n) => Array.from({ length: n }, (_, i) => i + 1)
                        .filter((i) => node.inputs?.find((inp) => inp.name === `${prefix}_${i}` && inp.link != null));
                    const vids = wired("ref_video", 3), vAud = wired("ref_video_audio", 3), auds = wired("ref_audio", 3);
                    if (vids.length || vAud.length || auds.length) {
                        const parts = [];
                        if (vids.length) parts.push(`video ${vids.map((i) => `<Video ${i}>`).join(", ")}`);
                        if (vAud.length) parts.push(`soundtrack for video ${vAud.join(", ")}`);
                        if (auds.length) parts.push(`${auds.length} standalone audio`);
                        container.appendChild(uiHint(`Also referenced via sockets: ${parts.join(" · ")}. Videos are described by the rewriter from sampled frames; audio is labelled <Audio j> but not described.`));
                    }
                }

                // Clip cards wrap into rows; validated clips collapse to one line each.
                const clipGrid = document.createElement("div");
                clipGrid.style.cssText = `
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 10px;
                    padding: 6px 2px 10px 2px;
                `;

                clipsState.forEach((clip, index) => {
                    const isValidated = Boolean(clip.validated);
                    if (isValidated && !expandedValidated.has(clip.id)) {
                        const row = document.createElement("div");
                        row.className = "minimax-clip-card";
                        row._clipId = clip.id;
                        row.title = "Validated. Click to expand.";
                        row.style.cssText = "grid-column: 1 / -1; display: flex; align-items: center; gap: 8px; background: #18261e; border: 1px solid #2e6a45; border-radius: 6px; padding: 5px 10px; cursor: pointer; box-sizing: border-box;";
                        const thumb = clipVideo(index, "width: 96px; height: 54px; flex: 0 0 96px;");
                        if (thumb) row.appendChild(thumb);
                        row.insertAdjacentHTML("beforeend", `
                            <span style="color: #4ade80; font-weight: 700; font-size: 12px; white-space: nowrap;">✓ Clip ${index + 1}</span>
                            <span style="background: #111118; color: #9c9cb8; padding: 1px 5px; border-radius: 3px; font-size: 10px; white-space: nowrap;">${clip.duration}s</span>
                            ${cacheBadge(index)}
                            <span style="flex: 1;"></span>
                        `);
                        row.appendChild(copyButton(clip));
                        row.insertAdjacentHTML("beforeend", `<span style="color: #6b8f7a; font-size: 11px;">▾</span>`);
                        row.onclick = () => { expandedValidated.add(clip.id); renderUI(); };
                        if (promptPanel.clipId === clip.id) markCardSelected(row, true);
                        clipGrid.appendChild(row);
                        return;
                    }
                    const card = document.createElement("div");
                    card.className = "minimax-clip-card";
                    card.style.cssText = `
                        min-width: 0;
                        background: ${isValidated ? "#18261e" : "#1c1c28"};
                        border: 1px solid ${isValidated ? "#2e6a45" : "#323246"};
                        border-radius: 7px;
                        padding: 10px;
                        display: flex;
                        flex-direction: column;
                        gap: 8px;
                        box-shadow: 0 3px 6px rgba(0,0,0,0.25);
                        box-sizing: border-box;
                    `;

                    const durValue = clampDuration(clip.duration || DUR_DEFAULT, clip.beyond);
                    const durMax = clip.beyond ? DUR_HARD_MAX : DUR_SOFT_MAX;
                    const frames = alignH3Frames(durValue);
                    const seedMode = clip.seed_mode || "randomize";

                    card.innerHTML = `
                        <!-- Card Header -->
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid ${isValidated ? '#285236' : '#28283a'}; padding-bottom: 6px;">
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <span style="font-weight: 700; color: ${isValidated ? "#4ade80" : "#ffffff"}; font-size: 13px;">Clip ${index + 1}</span>
                                <span style="background: #111118; color: #9c9cb8; padding: 1px 5px; border-radius: 3px; font-size: 10px;">${clip.duration}s (${frames}f)</span>
                                ${index > 0 ? `<span style="background: #2b2866; color: #c7d2fe; padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: 500;">🔗 Linked</span>` : ''}
                                ${externalPromptWired(index) ? `<span title="Prompt comes from the clip_prompt_${index + 1} input; the text below is ignored while it is connected." style="background: #1f3a2a; color: #86efac; padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: 500;">⇐ input</span>` : ''}
                                ${clip.planned === true ? `<span title="Planned from the film story by the story planner (auto_clips). Edit the text to make it yours; 'Replan from story' in the Prompt Rewriter section plans again." style="background: #1f2d3a; color: #7dd3fc; padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: 500;">✦ planned</span>` : ''}
                                ${!clip.prompt_rewritten && rewrittenTextOf(clip) ? `<span title="The original was edited after the last rewrite; the clip renders from the original until the next run rewrites it. The old rewrite is still readable in the prompt panel." style="background: #2a2438; color: #c4b5fd; padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: 500;">✎ pending</span>` : ''}
                                ${clip.prompt_rewritten ? `<span title="Rewritten by the built-in prompt rewriter${clip.rewrite_meta ? ` (${clip.rewrite_meta.model || ""}, ${clip.rewrite_meta.task || ""}${clip.rewrite_meta.thinking ? ", thinking" : ""}, ${clip.rewrite_meta.seconds || 0}s)` : ""}. It is not rewritten again; use 'Restore raw' in the prompt panel to redo it." style="background: #3a2e14; color: #fcd34d; padding: 1px 5px; border-radius: 3px; font-size: 9px; font-weight: 500;">✎ rewritten</span>` : ''}
                            </div>
                            <div style="display: flex; align-items: center; gap: 6px;">
                                <label style="display: flex; align-items: center; gap: 4px; cursor: pointer; color: ${isValidated ? "#4ade80" : "#aaa"}; font-size: 11px;">
                                    <input type="checkbox" class="val-check" ${isValidated ? "checked" : ""}>
                                    <span>Validated</span>
                                </label>
                                ${isValidated ? `<button class="fold-btn" title="Collapse this validated clip" style="background: transparent; border: none; color: #6b8f7a; cursor: pointer; font-size: 12px; padding: 0 2px;">▴</button>` : ''}
                                ${clipsState.length > 1 ? `<button class="del-btn" title="Delete Clip" style="background: transparent; border: none; color: #ef4444; cursor: pointer; font-size: 14px; padding: 0 2px;">✕</button>` : ''}
                            </div>
                        </div>

                        <!-- Prompt: compact preview; click it (or the card) to edit in the side panel -->
                        <div>
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 3px;">
                                <span style="display: flex; align-items: center; gap: 6px;"><span style="font-size: 10px; color: #8888a4;">Prompt:</span>${cacheBadge(index)}</span>
                                <span style="display: flex; align-items: center; gap: 6px;"><span class="prompt-meta" style="font-size: 10px; color: #6f6f8a; font-variant-numeric: tabular-nums;"></span><span class="copy-slot"></span></span>
                            </div>
                            <div class="prompt-preview minimax-prompt-preview" title="Click to edit in the prompt panel" style="box-sizing: border-box; width: 100%; min-height: ${PROMPT_PREVIEW_LINES * 15 + 12}px; max-height: ${PROMPT_PREVIEW_LINES * 15 + 12}px; overflow: hidden; background: #101016; border: 1px solid #2e2e42; border-radius: 4px; color: #d4d4e4; padding: 6px 8px; font-size: 11px; line-height: 15px; white-space: pre-wrap; word-break: break-word; cursor: text; position: relative;"></div>
                        </div>

                        <!-- Duration Setting: 5-15 s slider, "go beyond" unlocks up to 30 s -->
                        <div style="display: flex; flex-direction: column; gap: 4px; background: #13131c; padding: 4px 6px; border-radius: 4px; border: 1px solid #242434;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #9c9cb2; font-size: 11px;">Duration:</span>
                                <span class="dur-label" style="color: #fff; font-size: 11px; font-variant-numeric: tabular-nums;">${durValue}s</span>
                            </div>
                            <input type="range" class="dur-slider" value="${durValue}" step="1" min="${DUR_MIN}" max="${durMax}" style="width: 100%; accent-color: #6355d8; cursor: pointer;">
                            <label style="display: flex; align-items: center; gap: 5px; color: #8888a4; font-size: 10px; cursor: pointer;">
                                <input type="checkbox" class="dur-beyond" ${clip.beyond ? "checked" : ""} style="accent-color: #6355d8; margin: 0;">
                                go beyond ${DUR_SOFT_MAX}s (up to ${DUR_HARD_MAX}s, more VRAM)
                            </label>
                        </div>

                        <!-- Seed & Seed Mode Controls -->
                        <div style="display: flex; flex-direction: column; gap: 4px; background: #13131c; padding: 6px; border-radius: 4px; border: 1px solid #242434;">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <span style="color: #9c9cb2; font-size: 11px;">Seed Control:</span>
                                <select class="seed-mode" style="background: #1b1b26; border: 1px solid #36364e; border-radius: 3px; color: #9d8cff; font-size: 10px; padding: 2px 4px; cursor: pointer;">
                                    <option value="randomize" ${seedMode === "randomize" ? "selected" : ""}>randomize</option>
                                    <option value="fixed" ${seedMode === "fixed" ? "selected" : ""}>fixed</option>
                                    <option value="increment" ${seedMode === "increment" ? "selected" : ""}>increment</option>
                                    <option value="decrement" ${seedMode === "decrement" ? "selected" : ""}>decrement</option>
                                </select>
                            </div>
                            <div style="display: flex; gap: 4px; align-items: center;">
                                <input type="number" class="seed-input" value="${clip.seed || 42}" style="flex: 1; background: #1b1b26; border: 1px solid #36364e; border-radius: 3px; color: #fff; padding: 3px 6px; font-size: 11px;">
                                <button class="dice-btn" title="Roll Random Seed" style="background: #252536; border: 1px solid #3f3f58; color: #eee; border-radius: 3px; cursor: pointer; padding: 2px 8px; font-size: 12px;">🎲</button>
                            </div>
                        </div>
                    `;

                    // Event Listeners for Card
                    const preview = card.querySelector(".prompt-preview");
                    const promptMeta = card.querySelector(".prompt-meta");
                    const cardVideo = clipVideo(index, "width: 100%; aspect-ratio: 16 / 9;");
                    if (cardVideo) preview.parentElement.appendChild(cardVideo);
                    const paintPreview = () => {
                        const live = liveRewrite.get(clip.id);
                        if (cardVideo) {
                            cardVideo.style.display = live ? "none" : "block";
                            preview.style.display = live ? "" : "none";
                        }
                        const text = live ? (live.phase === "writing" ? live.text : clip.prompt || "") : (clip.prompt || "");
                        const { text: shown, fromSummary } = previewText(text);
                        preview.innerHTML = text.trim()
                            ? highlightH3(shown)
                            : `<span style="color: #55556e;">Enter clip prompt…</span>`;
                        preview.title = fromSummary ? "Showing the ## summary section. Click to edit the full prompt." : "Click to edit in the prompt panel";
                        if (externalPromptWired(index)) {
                            preview.style.opacity = "0.55";
                            preview.title = `Prompt is fed from the clip_prompt_${index + 1} input while it is connected; this text is ignored.`;
                            promptMeta.textContent = "from input" + (text.trim() ? ` · ${describePrompt(text)} (ignored)` : "");
                        } else {
                            preview.style.opacity = "1";
                        }
                        promptMeta.textContent = text.trim()
                            ? `${fromSummary ? "summary · " : ""}${describePrompt(text)}`
                            : "";
                        if (live) promptMeta.textContent = `✎ ${live.phase === "thinking" ? "thinking" : "writing"} · ${(live.text || "").length} chars`;
                        else if (!clip.prompt_rewritten && rewrittenTextOf(clip)) promptMeta.textContent = "pending rewrite · " + promptMeta.textContent;
                    };
                    paintPreview();
                    card._paintPreview = paintPreview;
                    card._clipId = clip.id;
                    if (promptPanel.clipId === clip.id) markCardSelected(card, true);
                    preview.onclick = () => openPromptPanel(clip.id);
                    card.addEventListener("click", (e) => {
                        if (e.target.closest("input, button, select, label, textarea")) return;
                        openPromptPanel(clip.id);
                    });

                    const durSlider = card.querySelector(".dur-slider");
                    const durLabel = card.querySelector(".dur-label");
                    const durBeyond = card.querySelector(".dur-beyond");
                    durSlider.oninput = () => {
                        durLabel.textContent = `${durSlider.value}s`;
                    };
                    durSlider.onchange = () => {
                        clip.duration = clampDuration(parseInt(durSlider.value, 10), clip.beyond);
                        saveState();
                        renderUI();
                    };
                    durBeyond.onchange = () => {
                        clip.beyond = durBeyond.checked;
                        clip.duration = clampDuration(clip.duration, clip.beyond);
                        saveState();
                        renderUI();
                    };

                    const seedInput = card.querySelector(".seed-input");
                    seedInput.onchange = () => {
                        clip.seed = parseInt(seedInput.value) || 42;
                        saveState();
                    };

                    const seedModeSelect = card.querySelector(".seed-mode");
                    seedModeSelect.onchange = () => {
                        clip.seed_mode = seedModeSelect.value;
                        saveState();
                    };

                    const diceBtn = card.querySelector(".dice-btn");
                    diceBtn.onclick = () => {
                        clip.seed = Math.floor(Math.random() * 1000000000);
                        seedInput.value = clip.seed;
                        saveState();
                    };

                    const valCheck = card.querySelector(".val-check");
                    valCheck.onchange = () => {
                        clip.validated = valCheck.checked;
                        expandedValidated.delete(clip.id);
                        saveState();
                        renderUI();
                    };

                    card.querySelector(".copy-slot").appendChild(copyButton(clip));
                    const foldBtn = card.querySelector(".fold-btn");
                    if (foldBtn) foldBtn.onclick = () => { expandedValidated.delete(clip.id); renderUI(); };

                    const delBtn = card.querySelector(".del-btn");
                    if (delBtn) {
                        delBtn.onclick = () => {
                            clipsState.splice(index, 1);
                            saveState();
                            renderUI();
                        };
                    }

                    clipGrid.appendChild(card);
                });

                container.appendChild(clipGrid);

                // 5. Footer: run mode + Run
                const footer = document.createElement("div");
                footer.style.cssText = "display: flex; gap: 10px; align-items: center;";
                const runRow = uiRow("run", bindSelect("run_mode", { render: (v) => String(v).replace("_", " ") }), { hint: "clip by clip renders one clip and pauses for validation; full batch renders everything; prompts only: write raw + H3 prompts, render nothing; edit them, then switch to full batch or clip by clip." });
                runRow.style.flex = "1";
                footer.appendChild(runRow);
                const continueButton = document.createElement("button");
                const runModeNow = getW("run_mode", "clip_by_clip");
                continueButton.textContent = runModeNow === "full_batch" ? "Run" : (runModeNow === "prompts_only" ? "Write prompts" : "Run / Continue after validation");
                continueButton.style.cssText = "flex: 1; background: #6355d8; color: #fff; border: none; border-radius: 6px; padding: 7px 12px; font-weight: 600; font-size: 12px; cursor: pointer;";
                continueButton.disabled = !clipsState.length || projectControls.isBusy();
                if (continueButton.disabled) continueButton.style.opacity = "0.5";
                continueButton.onclick = async () => {
                    saveState();
                    continueButton.disabled = true;
                    try { await app.queuePrompt(0, 1); }
                    finally { continueButton.disabled = false; }
                };
                footer.appendChild(continueButton);
                container.appendChild(footer);

                // 4. Progress Bar Area
                const progressBox = document.createElement("div");
                progressBox.id = "minimax-master-progress";
                progressBox.style.cssText = `
                    display: none;
                    background: #14141e;
                    border: 1px solid #333348;
                    border-radius: 6px;
                    padding: 8px 10px;
                    flex-direction: column;
                    gap: 4px;
                `;
                progressBox.innerHTML = `
                    <div style="display: flex; justify-content: space-between; font-size: 11px;">
                        <span id="master-progress-msg" style="color: #9d8cff; font-weight: 500;">Rendering...</span>
                        <span id="master-progress-pct" style="color: #aaa;">0%</span>
                    </div>
                    <div style="width: 100%; height: 6px; background: #222230; border-radius: 3px; overflow: hidden;">
                        <div id="master-progress-bar" style="width: 0%; height: 100%; background: linear-gradient(90deg, #6355d8, #9d8cff); transition: width 0.2s;"></div>
                    </div>
                `;
                container.appendChild(progressBox);

                if (promptPanel.el) {
                    const { clip } = findClip(promptPanel.clipId);
                    if (clip) openPromptPanel(clip.id, { focus: false }); else closePromptPanel();
                }
                fitNodeToContent();
            }

            renderUI();
            refreshCacheStatus();
            node.addDOMWidget("master_ui", "Master Director UI", container);

            // The panel draws every setting itself; the native widgets only hold
            // the values (and the API / saved-workflow contract).
            hideNativeWidgets(node);

            // Real-time WebSocket Progress Listener
            const progressHandler = (event) => {
                const data = event.detail;
                if (!data || String(data.owner) !== String(node.id)) return;

                const pBox = container.querySelector("#minimax-master-progress");
                const pMsg = container.querySelector("#master-progress-msg");
                const pPct = container.querySelector("#master-progress-pct");
                const pBar = container.querySelector("#master-progress-bar");

                if (pBox) {
                    pBox.style.display = "flex";
                    if (pMsg) pMsg.textContent = data.message || `Rendering clip ${data.clip_index + 1}...`;
                    const pct = Math.round((data.percent || 0.0) * 100);
                    if (pPct) pPct.textContent = `${pct}%`;
                    if (pBar) pBar.style.width = `${pct}%`;
                }
                if (data.stage === "done") refreshCacheStatus();
            };
            api.addEventListener(EVENT_PROGRESS, progressHandler);

            // The backend hands the clip list back (rewritten prompts, seeds used);
            // merge by clip id so the panel and the saved widget reflect the run.
            const clipsHandler = (event) => {
                const data = event.detail;
                if (!data || String(data.owner) !== String(node.id) || !Array.isArray(data.clips)) return;
                const byId = new Map(clipsState.map((c) => [c.id, c]));
                let changed = false;
                const keys = Array.isArray(data.fields) && data.fields.length
                    ? data.fields
                    : ["prompt", "prompt_raw", "prompt_rewritten", "rewrite_text", "rewrite_meta", "seed", "prompt_source"];
                // Clip ids repeat across workflows (0, 1, 2 ...), so only take an update
                // for a clip that is recognisably the one the server rendered: same raw
                // source for a rewrite pass, same prompt otherwise.
                const rawOf = (c) => (typeof c.prompt_raw === "string" && c.prompt_raw) || c.prompt || "";
                const sameClip = (local, incoming) => keys.includes("prompt")
                    ? rawOf(local) === rawOf(incoming) || (local.prompt || "") === (incoming.prompt_raw || incoming.prompt || "")
                    : (local.prompt || "") === (incoming.prompt || "");
                for (const incoming of data.clips) {
                    if (!incoming || typeof incoming !== "object") continue;
                    const local = byId.get(incoming.id);
                    if (!local || !sameClip(local, incoming)) continue;
                    for (const key of keys) {
                        if (key in incoming && JSON.stringify(local[key]) !== JSON.stringify(incoming[key])) {
                            local[key] = incoming[key];
                            changed = true;
                        }
                    }
                }
                if (!changed) return;
                saveState();
                renderUI();
                if (promptPanel.el) {
                    const { clip } = findClip(promptPanel.clipId);
                    if (clip) openPromptPanel(clip.id, { focus: false });
                }
            };
            api.addEventListener(EVENT_CLIPS, clipsHandler);

            // A story plan written into the clip list on the server: show it in the empty slots (never over a
            // clip that has text or already carries a planned flag) and add the clips past the end.
            const planHandler = (event) => {
                const data = event.detail;
                if (!data || String(data.owner) !== String(node.id) || !Array.isArray(data.planned)) return;
                let changed = false;
                for (const item of data.planned) {
                    const incoming = item?.clip;
                    if (!incoming || typeof incoming !== "object") continue;
                    const local = clipsState[item.index];
                    if (!local) {
                        if (item.index === clipsState.length) { clipsState.push(incoming); changed = true; }
                        continue;
                    }
                    if (rawTextOf(local).trim() || "planned" in local) continue;
                    // Clips imported from the Shot Writer arrive whole (seed, mode, title) and take the empty slot as they are.
                    if (incoming.from_writer === true) {
                        clipsState[item.index] = incoming;
                        changed = true;
                        continue;
                    }
                    Object.assign(local, { prompt: incoming.prompt, planned: true, duration: incoming.duration, prompt_rewritten: false, validated: false, plan_end_state: incoming.plan_end_state });
                    delete local.prompt_raw; delete local.rewrite_text; delete local.rewrite_meta;
                    changed = true;
                }
                if (!changed) return;
                ensureClipIds(clipsState);
                saveState();
                renderUI();
            };
            api.addEventListener(EVENT_PLAN, planHandler);

            // A clip's rewrite as it is being written: thinking, then the answer, then done.
            const rewriteHandler = (event) => {
                const data = event.detail;
                if (!data || String(data.owner) !== String(node.id)) return;
                const { clip } = findClip(data.clip_id);
                if (!clip) return;
                // Written from a raw text that was replaced after the run was queued:
                // it belongs to the old prompt, never apply it to the new one.
                if (typeof data.source === "string" && data.source.trim() !== rawTextOf(clip).trim()) {
                    if (liveRewrite.delete(clip.id)) renderUI();
                    return;
                }
                if (data.phase === "done") {
                    liveRewrite.delete(clip.id);
                    if (data.text) {
                        if (typeof clip.prompt_raw !== "string" || clip.prompt_rewritten === false) clip.prompt_raw = clip.prompt || "";
                        clip.rewrite_text = data.text;
                        clip.prompt = data.text;
                        clip.prompt_rewritten = true;
                        clip.validated = false;
                        saveState();
                    }
                    renderUI();
                } else {
                    liveRewrite.set(clip.id, { phase: data.phase, text: data.text || "" });
                    const card = [...container.querySelectorAll(".minimax-clip-card")].find((c) => c._clipId === clip.id);
                    card?._paintPreview?.();
                }
                if (promptPanel.el && promptPanel.clipId === clip.id) {
                    if (data.phase !== "done" && panelView(clip) !== "rewritten") { promptPanel.view = "rewritten"; promptPanel.viewClipId = clip.id; }
                    const stick = promptPanel.textarea && promptPanel.layer.scrollTop + promptPanel.layer.clientHeight >= promptPanel.layer.scrollHeight - 24;
                    openPromptPanel(clip.id, { focus: false });
                    if (stick && data.phase !== "done") {
                        promptPanel.layer.scrollTop = promptPanel.layer.scrollHeight;
                        promptPanel.textarea.scrollTop = promptPanel.textarea.scrollHeight;
                    }
                }
            };
            api.addEventListener(EVENT_REWRITE, rewriteHandler);
            const onRemoved = node.onRemoved;
            node.onRemoved = function () {
                api.removeEventListener(EVENT_PROGRESS, progressHandler);
                api.removeEventListener(EVENT_CLIPS, clipsHandler);
                api.removeEventListener(EVENT_PLAN, planHandler);
                api.removeEventListener(EVENT_REWRITE, rewriteHandler);
                return onRemoved?.apply(this, arguments);
            };
        };
    }
});

// The Shot Writer's last_output (what it wrote, saved in the video's metadata) is filled by the node and never edited: hidden, value and slot kept.
app.registerExtension({
    name: "MiniMaxH3.ShotWriterOutput",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "H3ShotWriter") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const widget = this.widgets?.find(w => w.name === "last_output");
            if (widget) {
                widget.type = "hidden";
                widget.computeSize = () => [0, -4];
            }
            return r;
        };
    }
});
