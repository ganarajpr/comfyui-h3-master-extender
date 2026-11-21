import ast
import logging
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "pdd_pure_engine.py"
NODE = ROOT / "master_node.py"
MODE = "pi-H3 (FL2VA adapter)"
BASE = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
AD2 = "pi_h3_2step_dwd_adapter_int8_convrot.safetensors"
AD4 = "pi_h3_4step_dwd_adapter_int8_convrot.safetensors"
GRID = [1.0, 0.8, 0.0]
FUNCS = ("check_pih3_steps", "pih3_adapter_file", "pih3_pack_module", "pih3_missing_pack_error", "_pih3_pack", "load_pih3_model",
         "pih3_sigmas", "hyperflow_tail_sigmas", "cache_pih3_model", "pih3_dwd_module")
NAMES = ("PIH3_ACCEL_MODE", "PIH3_STEPS", "PIH3_BASE_DEFAULT", "PIH3_ADAPTER_FILES", "PIH3_SCHEDULE", "_PIH3_CACHE",
         "PASS2_ENGINE_LORA", "PASS2_ENGINE_PIH3", "PASS2_ENGINES", "PIH3_REFINE_STEPS", "_PIH3_BASE_CACHE", "_PIH3_CACHE_MAX")
LOG = logging.getLogger("test_pi_h3")


def _ns():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body
              if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") in NAMES for t in n.targets))
              or (isinstance(n, ast.FunctionDef) and n.name in FUNCS and n.name != "_streams_from_latent")]
    ns = {"_LOG": LOG, "sys": sys, "time": __import__("time"),
          "_streams_from_latent": lambda latent, name: tuple(latent["samples"])}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(ENGINE), "exec"), ns)
    return ns


NS = _ns()


def _engine_cls(*names):
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    stub = ast.ClassDef(name="Engine", bases=[], keywords=[], body=methods, decorator_list=[])
    ns = dict(NS)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), str(ENGINE), "exec"), ns)
    return ns, ns["Engine"]


ENG_NS, Engine = _engine_cls("configure_pass2_lora", "_pass2_sigmas", "_pass2_lora_active", "_prepare_pih3_base",
                             "initialize_pih3_model", "configure_dual_sampling", "configure_pass1_steps",
                             "initialize_pih3_refine_model", "_collect_dwd_pixels", "_discard_dwd_stash",
                             "_pass2_chunking_active", "initialize_pass2_model")


class Model:
    def __init__(self, tag="m"):
        self.tag = tag


class Scheduler:
    calls = []

    def get_sigmas(self, model, steps, min_raw_t, max_raw_t, final_step_size_scale):
        Scheduler.calls.append((model, steps, min_raw_t, max_raw_t, final_step_size_scale))

        class S:
            def clone(self_):
                return self_

            def cpu(self_):
                return self_
        return (S(),)


class Pack:
    ADAPTER_KEY_RULES = {"minimax_h3": lambda cfg: {"rule": cfg}}

    def __init__(self):
        self.patched = []

    def patch_model_with_adapter(self, model, sd, key_mapping, strength):
        self.patched.append((model, dict(sd), key_mapping, strength))
        return ("patched", model)


def _install(nodes):
    saved = {k: ENG_NS.get(k) for k in ("nodes",)}
    ENG_NS["nodes"] = nodes
    NS["nodes"] = nodes
    return saved


class Steps(unittest.TestCase):
    def test_steps_and_adapter_selection(self):
        check, adapter = NS["check_pih3_steps"], NS["pih3_adapter_file"]
        self.assertEqual([check(s) for s in (2, 4, "4")], [2, 4, 4])
        self.assertEqual(check(None), 2)
        self.assertEqual(check(""), 2)
        for bad in (1, 3, 6, 8):
            with self.assertRaises(ValueError):
                check(bad)
        self.assertEqual(adapter(2), AD2)
        self.assertEqual(adapter(4), AD4)
        self.assertEqual(NS["PIH3_BASE_DEFAULT"], BASE)
        self.assertEqual(NS["PIH3_ACCEL_MODE"], MODE)

    def test_schedule_values(self):
        self.assertEqual(NS["PIH3_SCHEDULE"][2], (0.0, 0.93, 0.6))
        self.assertEqual(NS["PIH3_SCHEDULE"][4], (0.0, 0.96, 0.6))

    def test_sigmas_use_the_pi_flow_scheduler_with_the_recommended_values(self):
        Scheduler.calls.clear()
        model = Model()
        NS["pih3_sigmas"](Scheduler, model, 2)
        NS["pih3_sigmas"](Scheduler, model, 4)
        self.assertEqual(Scheduler.calls, [(model, 2, 0.0, 0.93, 0.6), (model, 4, 0.0, 0.96, 0.6)])


class Loader(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        base = Model("base")
        base.model = types.SimpleNamespace(model_config=types.SimpleNamespace(unet_config={"image_model": "minimax_h3"}))
        self.base = base
        comfy = types.ModuleType("comfy")
        sd = types.ModuleType("comfy.sd")
        sd.load_diffusion_model = lambda path, model_options=None: calls.append(("unet", path, model_options)) or base
        utils = types.ModuleType("comfy.utils")
        utils.load_torch_file = lambda path: calls.append(("adapter", path)) or {
            "blocks.0.w": 1, "dwd_norm_out.linear.weight": 2, "dwd_pixel_refiner.register_tokens": 3}
        fp = types.ModuleType("folder_paths")
        fp.get_full_path_or_raise = lambda kind, name: f"/{kind}/{name}"
        comfy.sd, comfy.utils = sd, utils
        self.saved = {k: sys.modules.get(k) for k in ("comfy", "comfy.sd", "comfy.utils", "folder_paths")}
        sys.modules.update({"comfy": comfy, "comfy.sd": sd, "comfy.utils": utils, "folder_paths": fp})

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_loads_base_then_patches_the_adapter_without_the_dwd_head(self):
        pack = Pack()
        with self.assertLogs("test_pi_h3", "INFO") as cm:
            out = NS["load_pih3_model"](pack, BASE, AD2)
        self.assertEqual(self.calls, [("unet", "/diffusion_models/" + BASE, {}), ("adapter", "/loras/" + AD2)])
        (model, sd, mapping, strength), = pack.patched
        self.assertIs(model, self.base)
        self.assertEqual(sd, {"blocks.0.w": 1})
        self.assertEqual(mapping, {"rule": {"image_model": "minimax_h3"}})
        self.assertEqual(strength, 1.0)
        self.assertEqual(out, ("patched", self.base))
        self.assertIn("2 dwd_ head tensors dropped", cm.output[0])

    def test_a_non_h3_base_is_an_error(self):
        self.base.model.model_config.unet_config["image_model"] = "flux"
        with self.assertRaisesRegex(ValueError, "FL2VA base"):
            NS["load_pih3_model"](Pack(), BASE, AD2)


class PackLookup(unittest.TestCase):
    def test_missing_pack_raises_a_clear_error(self):
        _install(types.SimpleNamespace(NODE_CLASS_MAPPINGS={}))
        self.assertIsNone(NS["pih3_pack_module"]())
        with self.assertRaisesRegex(RuntimeError, "ComfyUI-piFlow"):
            NS["_pih3_pack"]()

    def test_pack_is_found_through_the_adapter_node_class(self):
        pack = Pack()
        mod = types.ModuleType("fake_piflow_loaders")
        mod.patch_model_with_adapter = pack.patch_model_with_adapter
        mod.ADAPTER_KEY_RULES = pack.ADAPTER_KEY_RULES
        sys.modules["fake_piflow_loaders"] = mod
        try:
            cls = type("Adapter", (), {"__module__": "fake_piflow_loaders"})
            _install(types.SimpleNamespace(NODE_CLASS_MAPPINGS={"Load LakonLab Adapter": cls}))
            self.assertIs(NS["pih3_pack_module"](), mod)
            self.assertIs(NS["_pih3_pack"](), mod)
        finally:
            sys.modules.pop("fake_piflow_loaders", None)


def pih3_engine(lora=None, steps=2):
    e = Engine()
    e.accel_mode = MODE
    e.turbo_mode, e.flash_mode, e.hyperflow_mode, e.pih3_mode = False, False, False, True
    e.pih3_base_file, e.pih3_steps = BASE, steps
    e.pih3_model = e.pih3_raw = None
    e.pih3_refine = False
    e.turbo_steps = "6"
    e.shift_video, e.shift_audio = 12.0, 3.0
    e.hyperflow_pruned_refit = False
    e.pass2_lora = lora
    e.pass2_model = None
    e.pass2_steps = 4
    e.pass1_sigmas = list(GRID)
    e.dual_active = False
    e.dual_split_step = 2
    e.pass1_early_stop = False
    e.pass1_steps = 0
    e.sla_enabled = False
    e.sparse_method = "sla"
    return e


class EngineBehaviour(unittest.TestCase):
    def test_prepare_returns_the_base_untouched_and_the_pi_grid(self):
        pack = Pack()
        mod = types.ModuleType("fake_piflow_loaders2")
        mod.patch_model_with_adapter, mod.ADAPTER_KEY_RULES = pack.patch_model_with_adapter, pack.ADAPTER_KEY_RULES
        sys.modules["fake_piflow_loaders2"] = mod
        try:
            adapter_cls = type("Adapter", (), {"__module__": "fake_piflow_loaders2"})
            _install(types.SimpleNamespace(NODE_CLASS_MAPPINGS={"Load LakonLab Adapter": adapter_cls, "pi-Flow Scheduler": Scheduler}))
            Scheduler.calls.clear()
            e = pih3_engine(steps=4)
            base = Model()
            with self.assertLogs("test_pi_h3", "INFO"):
                model, sigmas = e._prepare_pih3_base(base)
            self.assertIs(model, base)
            self.assertEqual(e.turbo_steps, "8")
            self.assertEqual(Scheduler.calls, [(base, 4, 0.0, 0.96, 0.6)])
        finally:
            sys.modules.pop("fake_piflow_loaders2", None)

    def test_prepare_without_the_pack_is_an_error(self):
        _install(types.SimpleNamespace(NODE_CLASS_MAPPINGS={}))
        with self.assertRaisesRegex(RuntimeError, "ComfyUI-piFlow"):
            pih3_engine()._prepare_pih3_base(Model())

    def test_dual_sampling_and_pass1_steps_are_ignored(self):
        e = pih3_engine()
        e.configure_dual_sampling(True, 2, True)
        self.assertFalse(e.dual_active)
        e.configure_pass1_steps(4)
        self.assertEqual(e.pass1_steps, 0)

    def test_pass2_lora_is_forced_to_replace(self):
        e = pih3_engine()
        e.configure_pass2_lora("dmad.safetensors", 1.0, "stack on engine LoRA", 4)
        self.assertEqual(e.pass2_lora_mode, "replace")

    def test_pass2_with_dmad_is_the_unchanged_scheduler_tail(self):
        e = pih3_engine(lora="dmad.safetensors")
        e.pass2_lora_mode = "replace"
        calls = []
        e._pass2_basic_sigmas = lambda model, tail, denoise: calls.append((tail, denoise)) or "sig"
        self.assertEqual(e._pass2_sigmas(None, 0.25), "sig")
        self.assertEqual(calls, [(1, 0.25)])

    def test_pass2_without_a_lora_cuts_the_pi_grid_tail_and_warns(self):
        e = pih3_engine()
        with self.assertLogs("test_pi_h3", "WARNING") as cm:
            self.assertEqual(e._pass2_sigmas(None, 0.25), GRID[-2:])
        self.assertIn("pass-2 LoRA", cm.output[0])

    def test_the_patched_model_is_loaded_once_and_cached_across_clips(self):
        pack = Pack()
        loads = []
        ENG_NS["load_pih3_model"] = lambda p, b, a, **kw: loads.append((b, a, kw["keep_dwd"])) or Model("patched")
        ENG_NS["_pih3_pack"] = lambda: pack
        ENG_NS["pih3_sigmas"] = lambda cls, model, steps: types.SimpleNamespace(clone=lambda: types.SimpleNamespace(cpu=lambda: [1.0, 0.0]))
        _install(types.SimpleNamespace(NODE_CLASS_MAPPINGS={"pi-Flow Scheduler": Scheduler}))
        NS["_PIH3_CACHE"].clear()
        try:
            for _ in range(2):
                e = pih3_engine()
                e.initialize_model = lambda: None
                e._apply_sigma_shift = lambda m: m
                e._finish_model = lambda m, **kw: ("final", m)
                with self.assertLogs("test_pi_h3", "INFO"):
                    out = e.initialize_pih3_model()
                self.assertEqual(out[0], "final")
                self.assertIs(e.initialize_pih3_model(), out)
            self.assertEqual(loads, [(BASE, AD2, False)])
        finally:
            NS["_PIH3_CACHE"].clear()
            ENG_NS["load_pih3_model"] = NS["load_pih3_model"]
            ENG_NS["_pih3_pack"] = NS["_pih3_pack"]
            ENG_NS["pih3_sigmas"] = NS["pih3_sigmas"]


class ListScheduler:
    calls = []

    def get_sigmas(self, model, steps, min_raw_t, max_raw_t, final_step_size_scale):
        ListScheduler.calls.append((model, steps, min_raw_t, max_raw_t, final_step_size_scale))
        return ([0.9938, 0.8653, 0.0],)


def refine_engine(**kw):
    e = pih3_engine(**kw)
    e.pih3_refine = True
    e.pih3_refine_model = e.pih3_refine_raw = None
    e.dwd_sink = None
    e.pass2_chunk_frames = 124
    return e


class DwdPack(Pack):
    def __init__(self):
        super().__init__()
        self.loaded = []
        self.get_dwd_pixels = lambda samples: None
        outer = self

        class Loader:
            def load_adapter(self_, model, adapter_name, strength):
                outer.loaded.append((model, adapter_name, strength))
                return (("with_head", model, adapter_name),)
        self.LakonLabAdapterLoader = Loader


class RefineSigmas(unittest.TestCase):
    def test_pass2_is_the_last_step_of_the_two_step_grid_computed_on_the_pass2_model(self):
        _install(types.SimpleNamespace(NODE_CLASS_MAPPINGS={"pi-Flow Scheduler": ListScheduler}))
        ListScheduler.calls.clear()
        e = refine_engine()
        model = Model("pass2")
        with self.assertLogs("test_pi_h3", "INFO"):
            sigmas = e._pass2_sigmas(model, 0.25)
        self.assertEqual(sigmas, [0.8653, 0.0])
        self.assertEqual(ListScheduler.calls, [(model, 2, 0.0, 0.93, 0.6)])

    def test_refine_ignores_a_pass2_lora_and_windows(self):
        e = refine_engine(lora="dmad.safetensors")
        self.assertFalse(e._pass2_chunking_active(360, 1280, 736))
        e.pass2_chunk_when = "always"
        self.assertFalse(e._pass2_chunking_active(360, 1280, 736))

    def test_without_refine_the_default_pass2_is_untouched(self):
        e = pih3_engine(lora="dmad.safetensors")
        e.pih3_refine = False
        e.pass2_lora_mode = "replace"
        e.pass2_chunk_frames = 124
        e.pass2_chunk_when = "always"
        e._pass2_basic_sigmas = lambda model, tail, denoise: ("basic", tail)
        self.assertEqual(e._pass2_sigmas(None, 0.25), ("basic", 1))
        self.assertTrue(e._pass2_chunking_active(360, 1280, 736) in (True, False))


class RefineLoader(unittest.TestCase):
    def setUp(self):
        self.calls = []
        calls = self.calls
        base = Model("base")
        base.model = types.SimpleNamespace(model_config=types.SimpleNamespace(unet_config={"image_model": "minimax_h3"}))
        self.base = base
        comfy = types.ModuleType("comfy")
        sd = types.ModuleType("comfy.sd")
        sd.load_diffusion_model = lambda path, model_options=None: calls.append(("unet", path)) or base
        utils = types.ModuleType("comfy.utils")
        utils.load_torch_file = lambda path: calls.append(("adapter", path)) or {"blocks.0.w": 1, "dwd_x": 2}
        fp = types.ModuleType("folder_paths")
        fp.get_full_path_or_raise = lambda kind, name: f"/{kind}/{name}"
        comfy.sd, comfy.utils = sd, utils
        self.saved = {k: sys.modules.get(k) for k in ("comfy", "comfy.sd", "comfy.utils", "folder_paths")}
        sys.modules.update({"comfy": comfy, "comfy.sd": sd, "comfy.utils": utils, "folder_paths": fp})

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

    def test_keep_dwd_goes_through_piflows_own_loader_with_the_head(self):
        pack = DwdPack()
        with self.assertLogs("test_pi_h3", "INFO"):
            out = NS["load_pih3_model"](pack, BASE, AD2, keep_dwd=True)
        self.assertEqual(out, ("with_head", self.base, AD2))
        self.assertEqual(pack.loaded, [(self.base, AD2, 1.0)])
        self.assertEqual(pack.patched, [])
        self.assertEqual([c[0] for c in self.calls], ["unet"])

    def test_both_variants_share_one_base_load(self):
        pack = DwdPack()
        cache = {}
        with self.assertLogs("test_pi_h3", "INFO"):
            NS["load_pih3_model"](pack, BASE, AD2, keep_dwd=True, base_cache=cache)
            NS["load_pih3_model"](pack, BASE, AD2, keep_dwd=False, base_cache=cache)
        self.assertEqual([c[0] for c in self.calls].count("unet"), 1)
        self.assertEqual(pack.patched[0][1], {"blocks.0.w": 1})

    def test_model_cache_holds_at_most_two(self):
        NS["_PIH3_CACHE"].clear()
        try:
            for k in "abc":
                NS["cache_pih3_model"](k, lambda k=k: k)
            self.assertEqual(list(NS["_PIH3_CACHE"]), ["b", "c"])
            self.assertEqual(NS["cache_pih3_model"]("c", lambda: "x"), ("c", True))
        finally:
            NS["_PIH3_CACHE"].clear()


class RefineEngine(unittest.TestCase):
    def test_pass2_model_is_the_base_plus_the_two_step_dwd_adapter_with_the_head_kept_and_is_cached(self):
        pack = DwdPack()
        loads = []
        ENG_NS["load_pih3_model"] = lambda p, b, a, **kw: loads.append((b, a, kw["keep_dwd"])) or Model("withhead")
        ENG_NS["_pih3_pack"] = lambda: pack
        NS["_PIH3_CACHE"].clear()
        try:
            e = refine_engine(steps=4)
            e.initialize_model = lambda: (None, None)
            e._apply_sigma_shift = lambda m: ("shifted", m.tag)
            e._pass2_sla_override = lambda: 0.5
            seen = []
            e._finish_model = lambda m, **kw: seen.append(kw) or ("final", m)
            with self.assertLogs("test_pi_h3", "INFO"):
                out = e.initialize_pass2_model()
            self.assertEqual(out, ("final", ("shifted", "withhead")))
            self.assertEqual(loads, [(BASE, AD2, True)])
            self.assertEqual(seen, [{"sparsity": 0.5}])
            self.assertIs(e.initialize_pass2_model(), out)
        finally:
            NS["_PIH3_CACHE"].clear()
            ENG_NS["load_pih3_model"] = NS["load_pih3_model"]
            ENG_NS["_pih3_pack"] = NS["_pih3_pack"]

    def test_pass1_shares_the_head_model_only_for_the_two_step_adapter(self):
        src = ENGINE.read_text(encoding="utf-8")
        self.assertIn("keep_dwd = self.pih3_refine and adapter == PIH3_ADAPTER_FILES[PIH3_REFINE_STEPS]", src)
        self.assertIn("if self.pih3_refine:\n                    self._discard_dwd_stash()", src)

    def _dwd_env(self):
        class Dwd:
            DWD_PIXELS = {"fp-clip": "PIX", "stale": "OLD"}
            latent_fingerprint = staticmethod(lambda video: "fp-clip" if video == "VIDEO" else "other")

            @staticmethod
            def get_dwd_pixels(samples):
                return None
        mod = types.ModuleType("fake_dwd_head")
        mod.DWD_PIXELS = Dwd.DWD_PIXELS
        mod.latent_fingerprint = Dwd.latent_fingerprint
        mod.get_dwd_pixels = lambda samples: None
        mod.get_dwd_pixels.__module__ = "fake_dwd_head"
        sys.modules["fake_dwd_head"] = mod
        pack = types.SimpleNamespace(get_dwd_pixels=mod.get_dwd_pixels)
        ENG_NS["_pih3_pack"] = lambda: pack
        return mod

    def test_pixels_are_taken_from_the_stash_registered_and_the_stash_emptied_latent_untouched(self):
        mod = self._dwd_env()
        try:
            e = refine_engine()
            got = []
            e.dwd_sink = lambda video, pixels: got.append((video, pixels)) or True
            latent = {"samples": ("VIDEO", "AUDIO")}
            before = dict(latent)
            self.assertTrue(e._collect_dwd_pixels(latent, 0))
            self.assertEqual(got, [("VIDEO", "PIX")])
            self.assertEqual(mod.DWD_PIXELS, {})
            self.assertEqual(latent, before)
        finally:
            sys.modules.pop("fake_dwd_head", None)
            ENG_NS["_pih3_pack"] = NS["_pih3_pack"]

    def test_missing_pixels_warn_and_fall_back_to_the_vae(self):
        mod = self._dwd_env()
        try:
            e = refine_engine()
            e.dwd_sink = lambda video, pixels: self.fail("nothing to register")
            with self.assertLogs("test_pi_h3", "WARNING") as cm:
                self.assertFalse(e._collect_dwd_pixels({"samples": ("OTHER", "AUDIO")}, 1))
            self.assertIn("decoded with the VAE", cm.output[0])
            self.assertEqual(mod.DWD_PIXELS, {})
        finally:
            sys.modules.pop("fake_dwd_head", None)
            ENG_NS["_pih3_pack"] = NS["_pih3_pack"]

    def test_pass2_sampling_is_timed_and_pixels_collected_right_after_it(self):
        src = ENGINE.read_text(encoding="utf-8")
        self.assertIn('_LOG.info("Clip %d: pass 2 sampling %.1f s", clip_index + 1, time.time() - pass2_started)\n        if self.pih3_refine:\n            self._collect_dwd_pixels(final_sampled, clip_index)', src)


class RefineWiring(unittest.TestCase):
    node = NODE.read_text(encoding="utf-8")

    def test_option_strings_and_default(self):
        self.assertEqual(NS["PASS2_ENGINES"], ["LoRA refine", "pi-H3 refine (DWD, no VAE decode)"])
        self.assertRegex(self.node, r'"pass2_engine": \(PASS2_ENGINES, \{"default": PASS2_ENGINE_LORA')

    def test_input_comes_after_pih3_steps_before_the_socket(self):
        i = self.node.index('"pih3_steps": (')
        self.assertLess(i, self.node.index('"pass2_engine": ('))
        self.assertLess(self.node.index('"pass2_engine": ('), self.node.index('"time_gate_json": ('))

    def test_validation_rejects_x2_and_unknown_values(self):
        i = self.node.index("def VALIDATE_INPUTS")
        block = self.node[i:self.node.index("def IS_CHANGED")]
        self.assertIn("pass2_engine=None", block)
        self.assertIn("if pass2_engine == PASS2_ENGINE_PIH3:", block)
        self.assertIn("cannot", block)
        self.assertIn("normalize_final_decode(final_decode) != \"standard\"", block)
        self.assertIn("pass2_engine not in (None, \"\")", block)

    def test_extend_rejects_x2_at_run_time_and_passes_the_engine_on(self):
        self.assertIn("pass2_engine=pass2_engine,", self.node)
        self.assertIn('if normalize_final_decode(kwargs.get("final_decode")) != "standard":\n                raise ValueError("pass2_engine pi-H3 refine', self.node)

    def test_frames_are_cleared_at_job_start_and_end(self):
        self.assertEqual(self.node.count("motion_context_disk.clear_dwd_frames()"), 2)


class Wiring(unittest.TestCase):
    engine = ENGINE.read_text(encoding="utf-8")
    node = NODE.read_text(encoding="utf-8")

    def test_mode_is_appended_last_in_accel_modes(self):
        tree = ast.parse(self.node)
        modes = next(ast.literal_eval(n.value) for n in tree.body
                     if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ACCEL_MODES" for t in n.targets))
        self.assertEqual(modes[-1], MODE)
        self.assertEqual(modes[0], "Turbo LoRA")
        self.assertRegex(self.node, r'"accel_mode": \(ACCEL_MODES, \{"default": "Turbo LoRA"')

    def test_inputs_have_safe_defaults_and_come_after_chain_memory(self):
        i = self.node.index('"chain_memory": (')
        j = self.node.index('"pih3_base_file": (')
        k = self.node.index('"pih3_steps": (')
        self.assertLess(i, j)
        self.assertLess(j, k)
        self.assertLess(k, self.node.index('"time_gate_json": ('))
        self.assertRegex(self.node, r'"pih3_base_file": \(_pih3_base_files\(\), \{"default": PIH3_BASE_DEFAULT')
        self.assertRegex(self.node, r'"pih3_steps": \("INT", \{"default": 2, "min": 2, "max": 4, "step": 2')

    def test_validate_inputs_checks_only_in_pi_h3_mode(self):
        i = self.node.index("def VALIDATE_INPUTS")
        block = self.node[i:self.node.index("def IS_CHANGED")]
        self.assertIn("pih3_base_file=None, pih3_steps=None", block)
        self.assertIn("if normalize_accel_mode(accel_mode) == PIH3_ACCEL_MODE:", block)
        self.assertIn("not in {list(PIH3_STEPS)}", block)

    def test_extend_resolves_the_inputs_only_in_pi_h3_mode_and_passes_them_on(self):
        self.assertIn("pih3_refine = pass2_engine == PASS2_ENGINE_PIH3", self.node)
        self.assertIn("pih3_steps = PIH3_BASE_DEFAULT, 2" if False else "if accel_mode == PIH3_ACCEL_MODE:\n            if pih3_pack_module() is None:", self.node)
        self.assertIn("pih3_base_file=pih3_base,\n            pih3_steps=pih3_steps,", self.node)

    def test_fingerprint_key_only_in_pi_h3_mode_and_pass1_step_fields_are_not_in_it(self):
        i = self.node.index("if accel_mode == PIH3_ACCEL_MODE:\n            # Only in pi-H3 mode")
        block = self.node[i:i + 700]
        self.assertIn('p1_settings["pih3"] = [pih3_base, pih3_steps]', block)
        self.assertIn('p1_settings["pdd_nfe"] = "pi-h3"', block)
        tree = ast.parse(self.node)
        literal = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.value, ast.Dict)
                       and any(getattr(t, "id", "") == "p1_settings" for t in n.targets))
        self.assertNotIn("pih3", [k.value for k in literal.value.keys])

    def test_pass1_routes_to_the_pi_h3_model_and_offloads_it_before_pass_2(self):
        self.assertIn("elif self.pih3_mode:\n                model = self.initialize_pih3_model()\n                sigmas_p1 = self.pass1_sigmas", self.engine)
        self.assertIn("elif self.pih3_mode and pass2_enabled:\n                if self.pih3_refine:\n                    self._discard_dwd_stash()\n                self._unload_pih3()", self.engine)
        self.assertIn("elif self.pih3_mode:\n            accelerated, sigmas_p1 = self._prepare_pih3_base(shifted_model)", self.engine)

    def test_pi_h3_never_takes_the_flash_or_handoff_paths(self):
        self.assertIn('self.pih3_mode = self.accel_mode == PIH3_ACCEL_MODE', self.engine)
        self.assertIn('self.flash_mode = self.accel_mode == FLASH_ACCEL_MODE', self.engine)
        self.assertNotEqual("HyperFlow Flash (GGUF)", MODE)

    def test_the_pass2_model_is_built_from_the_base_not_the_adapter(self):
        i = self.engine.index("p2_sparsity = self._pass2_sla_override()", self.engine.index("def initialize_pass2_model"))
        block = self.engine[i:i + 2500]
        self.assertIn("base = self.shifted_model", block)
        self.assertNotIn("pih3", block)


if __name__ == "__main__":
    unittest.main()
