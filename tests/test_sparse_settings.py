import ast
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "pdd_pure_engine.py"


def _load():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = [n for n in tree.body
              if (isinstance(n, ast.FunctionDef) and n.name == "resolve_sparse_settings")
              or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "SPARSE_SINK_CHOICES" for t in n.targets))]
    ns = {}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns["resolve_sparse_settings"]


resolve = _load()
AUTO = (-1.0, "auto", "auto")


class SparseSettingsTests(unittest.TestCase):
    def test_hyperflow_sol_attn_auto_is_the_recipe(self):
        r = resolve(True, "sol-attn", *AUTO)
        self.assertEqual((r["start"], r["dense_blocks"], r["sink"]), (0.16, "0,1", "off"))
        self.assertEqual(set(r["sources"].values()), {"hyperflow recipe"})

    def test_everything_else_keeps_the_old_values(self):
        for hyperflow, method in ((False, "sol-attn"), (True, "sla"), (True, "vsa"), (False, "sla")):
            r = resolve(hyperflow, method, *AUTO)
            self.assertEqual((r["start"], r["dense_blocks"], r["sink"]), (0.2, "", "exact_kv_and_rows"))
            self.assertEqual(set(r["sources"].values()), {"default"})

    def test_user_values_win_in_every_mode(self):
        for hyperflow, method in ((True, "sol-attn"), (False, "sla")):
            r = resolve(hyperflow, method, 0.3, "2,3", "exact_kv")
            self.assertEqual((r["start"], r["dense_blocks"], r["sink"]), (0.3, "2,3", "exact_kv"))
            self.assertEqual(set(r["sources"].values()), {"user"})

    def test_empty_dense_blocks_is_a_user_choice_and_zero_start_is_valid(self):
        r = resolve(True, "sol-attn", 0.0, "", "off")
        self.assertEqual((r["start"], r["dense_blocks"]), (0.0, ""))
        self.assertEqual(r["sources"]["dense_blocks"], "user")

    def test_junk_start_means_auto(self):
        self.assertEqual(resolve(False, "sla", "", "auto", "auto")["start"], 0.2)

    def test_default_start_follows_the_class_constant(self):
        self.assertEqual(resolve(False, "sla", *AUTO, default_start=0.25)["start"], 0.25)


if __name__ == "__main__":
    unittest.main()
