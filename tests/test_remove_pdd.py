import ast
import logging
import re
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "master_node.py"


def _load():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    wanted = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in ("is_legacy_accel_mode", "normalize_accel_mode"):
            wanted.append(node)
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "ACCEL_MODES" for t in node.targets):
            wanted.append(node)
    ns = {"_LOG": logging.getLogger("test_remove_pdd"), "HYPERFLOW_ACCEL_MODE": "HyperFlow 8-step"}
    exec(compile(ast.Module(body=wanted, type_ignores=[]), str(SOURCE), "exec"), ns)
    return ns


class RemovePDDTests(unittest.TestCase):
    ns = _load()

    def test_old_pdd_mode_runs_hyperflow(self):
        norm = self.ns["normalize_accel_mode"]
        with self.assertLogs("test_remove_pdd", level="WARNING"):
            self.assertEqual(norm("PDD 8-step"), "HyperFlow 8-step")

    def test_current_modes_are_untouched(self):
        norm = self.ns["normalize_accel_mode"]
        self.assertEqual(norm("HyperFlow 8-step"), "HyperFlow 8-step")
        self.assertEqual(norm("Turbo LoRA"), "Turbo LoRA")
        self.assertEqual(norm("HyperFlow Flash (GGUF)"), "HyperFlow Flash (GGUF)")
        self.assertEqual(norm("HyperFlow 8-step \u00b7 dual sampling"), "HyperFlow 8-step \u00b7 dual sampling")
        self.assertEqual(self.ns["ACCEL_MODES"], ["Turbo LoRA", "HyperFlow Flash (GGUF)", "Pika \u2192 Flash hand-off", "Base + CFG (undistilled)", "CFG \u2192 Pika hand-off",
                                                  "HyperFlow 8-step", "HyperFlow 8-step \u00b7 dual sampling", "pi-H3 (FL2VA adapter)"])

    def test_pdd_file_keeps_its_slot_and_accepts_any_string(self):
        text = SOURCE.read_text(encoding="utf-8")
        self.assertRegex(text, r'"pdd_file": \("STRING", \{"default": ""')
        names = re.findall(r'^\s{16}"(\w+)": \(', text[text.index("def INPUT_TYPES"):], re.M)
        self.assertEqual(names[:5], ["model", "clip", "vae", "audio_vae", "pdd_file"])
        self.assertEqual(names[names.index("pass2_denoise") + 1], "pdd_nfe")


if __name__ == "__main__":
    unittest.main()
