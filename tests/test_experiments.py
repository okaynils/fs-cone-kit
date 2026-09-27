import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from core.data.base import BaseDataset
from core.experiments import ExperimentArtifacts, collect_run_metadata, redact_sensitive


class FixtureDataset(BaseDataset):
    def _download(self):
        pass

    def _preprocess(self):
        pass


class ExperimentArtifactTests(unittest.TestCase):
    def test_sensitive_values_are_redacted(self):
        value = {
            "token": "do-not-write-me",
            "tracking_uri": "https://user:password@example.test/path",
            "callback_uri": "https://example.test/path?token=query-secret&safe=yes",
            "nested": {"api_key": "also-secret", "safe": 42},
        }
        redacted = redact_sensitive(value)
        self.assertEqual(redacted["token"], "<redacted>")
        self.assertNotIn("password", redacted["tracking_uri"])
        self.assertNotIn("query-secret", redacted["callback_uri"])
        self.assertIn("safe=yes", redacted["callback_uri"])
        self.assertEqual(redacted["nested"]["api_key"], "<redacted>")
        self.assertEqual(redacted["nested"]["safe"], 42)

    def test_command_metadata_redacts_secret_overrides(self):
        with patch("sys.argv", ["train", "logger.token=secret", "--password", "secret-2"]):
            metadata = collect_run_metadata(Path(__file__).resolve().parents[1], 42)
        command = " ".join(metadata["command"])
        self.assertNotIn("secret", command)

    def test_artifact_start_writes_config_metadata_and_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_info = {
                "fingerprint": "abc",
                "split_counts": {"train": 1},
                "manifest": {"splits": {"train": ["images/train/a.jpg"]}},
            }
            artifacts = ExperimentArtifacts(root)
            artifacts.start(
                {"seed": 7, "logger": {"password": "secret"}},
                dataset_info,
                Path(__file__).resolve().parents[1],
            )
            config = yaml.safe_load((root / "experiment/config.yaml").read_text())
            metadata = json.loads((root / "experiment/metadata.json").read_text())
            self.assertEqual(config["logger"]["password"], "<redacted>")
            self.assertEqual(metadata["seed"], 7)
            self.assertTrue((root / "experiment/splits.json").exists())

    def test_generic_yolo_dataset_gets_a_persistent_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images/train").mkdir(parents=True)
            (root / "images/val").mkdir(parents=True)
            (root / "images/train/a.jpg").touch()
            (root / "images/val/b.jpg").touch()
            (root / "dataset.yaml").write_text("train: images/train\nval: images/val\n")
            dataset = FixtureDataset("raw", str(root), {"cone": 0})
            dataset.prepare()
            info = dataset.get_dataset_info()
            self.assertEqual(info["split_counts"], {"train": 1, "val": 1})
            self.assertEqual(len(info["fingerprint"]), 64)

            (root / "images/train/c.jpg").touch()
            dataset.prepare()
            changed = dataset.get_dataset_info()
            self.assertEqual(changed["split_counts"]["train"], 2)
            self.assertNotEqual(changed["fingerprint"], info["fingerprint"])


if __name__ == "__main__":
    unittest.main()
