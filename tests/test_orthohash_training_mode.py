from __future__ import annotations

import ast
import inspect
import textwrap
import unittest

from baseline.OrthoHash import OrthoHash


class OrthoHashTrainingModeRegressionTest(unittest.TestCase):
    def test_epoch_loop_restores_train_mode_after_periodic_evaluation(self) -> None:
        source = textwrap.dedent(inspect.getsource(OrthoHash._train_model))
        tree = ast.parse(source)
        epoch_loops = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.target, ast.Name)
            and node.target.id == "epoch"
        ]
        self.assertEqual(len(epoch_loops), 1)
        calls_train = any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "model"
            and node.func.attr == "train"
            for node in ast.walk(epoch_loops[0])
        )
        self.assertTrue(
            calls_train,
            "OrthoHash must restore model.train() inside every epoch because "
            "periodic extraction leaves the wrapper in eval mode",
        )


if __name__ == "__main__":
    unittest.main()
