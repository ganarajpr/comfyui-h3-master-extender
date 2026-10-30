"""_replace_with_retry (motion_context_disk): a transient Windows lock on the destination must not fail a finished
render, and a lock that never clears must still raise. The function is extracted from the source, so this needs no
ComfyUI or torch."""
import ast
import types
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "motion_context_disk.py"


def _load(fake_os, sleeps):
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_replace_with_retry")
    ns = {"os": fake_os, "time": types.SimpleNamespace(sleep=sleeps.append)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(SRC), "exec"), ns)
    return ns["_replace_with_retry"]


class ReplaceWithRetry(unittest.TestCase):
    def test_transient_lock_is_retried(self):
        calls = []

        def replace(src, dst):
            calls.append((src, dst))
            if len(calls) < 3:
                raise PermissionError(5, "Access is denied")

        sleeps = []
        _load(types.SimpleNamespace(replace=replace), sleeps)("a.tmp", "a.json")
        self.assertEqual(len(calls), 3)
        self.assertEqual(sleeps, [0.02, 0.04])

    def test_permanent_lock_still_raises(self):
        def replace(src, dst):
            raise PermissionError(5, "Access is denied")

        sleeps = []
        with self.assertRaises(PermissionError):
            _load(types.SimpleNamespace(replace=replace), sleeps)("a.tmp", "a.json")
        self.assertEqual(len(sleeps), 11)
        self.assertLess(sum(sleeps), 5.0)

    def test_other_errors_are_not_swallowed(self):
        def replace(src, dst):
            raise FileNotFoundError(2, "gone")

        with self.assertRaises(FileNotFoundError):
            _load(types.SimpleNamespace(replace=replace), [])("a.tmp", "a.json")


if __name__ == "__main__":
    unittest.main()
