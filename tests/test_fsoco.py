import unittest
from pathlib import Path

from core.data.fsoco import FSOCODataset


def dataset(seed: int) -> FSOCODataset:
    return FSOCODataset(
        download_url="https://example.test/fsoco.zip",
        raw_dir="raw",
        preprocessed_dir="prepared",
        class_map={"blue_cone": 0},
        split_seed=seed,
        val_fraction=0.2,
        test_fraction=0.1,
    )


class FSOCOSplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_has_expected_sizes(self):
        annotations = [Path(f"team/ann/{index}.json") for index in range(100)]
        first = dataset(42)._split_annotations(annotations)
        repeated = dataset(42)._split_annotations(list(reversed(annotations)))

        self.assertEqual(first, repeated)
        self.assertEqual({name: len(paths) for name, paths in first.items()}, {
            "test": 10,
            "val": 20,
            "train": 70,
        })
        self.assertEqual(len(set().union(*map(set, first.values()))), 100)

    def test_seed_changes_membership(self):
        annotations = [Path(f"team/ann/{index}.json") for index in range(100)]
        self.assertNotEqual(
            dataset(42)._split_annotations(annotations),
            dataset(43)._split_annotations(annotations),
        )


if __name__ == "__main__":
    unittest.main()
