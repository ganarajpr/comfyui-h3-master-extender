import ast
import logging
import types
import unittest
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1] / "pdd_pure_engine.py"
METHODS = ("initialize_pass2_model", "_pass2_sla_override", "_apply_sla", "_finish_model", "_pass2_lora_active")


class Model:
    def __init__(self, tag="base"):
        self.tag = tag

    def clone(self):
        return Model(self.tag)

    def set_model_optimized_attention(self, fn):
        pass


class Native:
    calls = []

    @classmethod
    def execute(cls, model, selection, **kw):
        cls.calls.append(dict(selection))
        out = Model(model.tag)
        out.keep = selection.get("keep_percent")
        return (out,)


def _engine_class():
    tree = ast.parse(ENGINE.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MasterEngine")
    body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    body += [n for n in cls.body if isinstance(n, ast.Assign) and any(getattr(t, "id", "") in ("SPARSE_MIN_TOKENS", "SPARSE_EXTRA_TOKENS") for t in n.targets)]
    stub = ast.ClassDef(name="Engine", bases=[], keywords=[], body=body, decorator_list=[])
    ns = {
        "_LOG": logging.getLogger("test_pass2_sla"),
        "nodes": types.SimpleNamespace(NODE_CLASS_MAPPINGS={"BlockSparseAttention": Native, "LoraLoaderModelOnly": Lora}),
        "_safe_get_output": lambda res, i, name: res[i],
        "get_attention_function": lambda name, _: object(),
    }
    exec(compile(ast.fix_missing_locations(ast.Module(body=[stub], type_ignores=[])), str(ENGINE), "exec"), ns)
    return ns["Engine"]


class Lora:
    def load_lora_model_only(self, model, name, strength):
        return (Model(model.tag + "+lora"),)


Engine = _engine_class()


def make(p2=0.0, lora=False, method="sla", sla=0.9):
    e = Engine()
    e.sla_enabled = True
    e.sla_sparsity = sla
    e.pass2_sla_sparsity = p2
    e.sparse_method = method
    e.sparse_tau = 1.3
    e.attention_backend = "pytorch attention"
    e.hyperflow_mode = True
    e.flash_mode = False
    e.sparse_start_percent, e.sparse_dense_blocks, e.sparse_sink = -1.0, "auto", "auto"
    e._sparse_settings = lambda: {"start": 0.2, "dense_blocks": "", "sink": "off"}
    e.shifted_model = Model("shifted")
    e.accel_model = Model("accel")
    e.pass2_model = None
    e.pass2_sla_model = None
    e.pass2_lora = "x" if lora else None
    e.pass2_lora_strength = 1.0
    e.pass2_lora_mode = "replace"
    e.pass2_steps = 4
    e.turbo_steps = 8
    e.prepared_model = None
    e.initialize_model = lambda: (e.pass1_model(), None)
    e.pass1_model = lambda: e._finish_model(e.accel_model)
    return e


class Pass2SlaTests(unittest.TestCase):
    def setUp(self):
        Native.calls.clear()

    def test_default_pass2_uses_pass1_model_unchanged(self):
        e = make(0.0)
        m = e.initialize_pass2_model()
        self.assertAlmostEqual(m.keep, 10.0)
        self.assertEqual(len(Native.calls), 1)

    def test_override_without_lora_repatches_from_unsparsed_accel_model(self):
        e = make(0.95)
        p1 = e.pass1_model()
        p2 = e.initialize_pass2_model()
        self.assertAlmostEqual(p1.keep, 10.0)
        self.assertAlmostEqual(p2.keep, 5.0)
        self.assertEqual(e.sla_sparsity, 0.9)
        self.assertIs(e.initialize_pass2_model(), p2)

    def test_override_with_lora_model(self):
        e = make(0.95, lora=True)
        p2 = e.initialize_pass2_model()
        self.assertEqual(p2.tag, "shifted+lora")
        self.assertAlmostEqual(p2.keep, 5.0)

    def test_next_clip_pass1_is_back_to_base_sparsity(self):
        e = make(0.95)
        e.initialize_pass2_model()
        p1_next = e.pass1_model()
        self.assertAlmostEqual(p1_next.keep, 10.0)

    def test_sol_attn_ignores_override(self):
        e = make(0.95, method="sol-attn")
        e.initialize_pass2_model()
        self.assertTrue(all("keep_percent" not in c for c in Native.calls))

    def test_disabled_sla_ignores_override(self):
        e = make(0.95)
        e.sla_enabled = False
        self.assertIsNone(e._pass2_sla_override())


if __name__ == "__main__":
    unittest.main()
