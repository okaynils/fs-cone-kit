import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from core.train import run


class TrainingSafetyTests(unittest.TestCase):
    def test_existing_checkpoint_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary)
            weights = experiment / "ultralytics_files/weights"
            weights.mkdir(parents=True)
            (weights / "last.pt").touch()
            cfg = OmegaConf.create({"trainer": {"resume_from": None}})
            with self.assertRaisesRegex(FileExistsError, "resume"):
                run(cfg, experiment)


if __name__ == "__main__":
    unittest.main()
