"""CPU-only artifact smoke test. It uses tiny records and performs no downloads."""

import tempfile
import unittest
from pathlib import Path

from core.comparison import collect_comparison_rows, write_comparison
from tests.test_comparison import create_experiment


class CpuSmokeTest(unittest.TestCase):
    def test_two_tiny_experiments_produce_a_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiments = [
                create_experiment(root, "tiny-n", "fixture-split", 0.1),
                create_experiment(root, "tiny-s", "fixture-split", 0.2),
            ]
            rows = collect_comparison_rows(experiments)
            csv_path, markdown_path = write_comparison(rows, root / "comparison")
            self.assertTrue(csv_path.exists())
            self.assertTrue(markdown_path.exists())


if __name__ == "__main__":
    unittest.main()
