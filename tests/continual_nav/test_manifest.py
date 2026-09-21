from pathlib import Path
import tempfile
import unittest

from fixtures import map_file
from tasks.continual_nav.data.manifest import build_manifest, MazeManifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for i in range(5):
            map_file(self.root, i)
        for i in range(2):
            map_file(self.root, i, "test")

    def build(self):
        return build_manifest(self.root, validation_count=2, validation_episodes=2,
                              test_episodes=2, drift_episodes=1)

    def test_deterministic_disjoint_splits_and_ignored_root_files(self):
        (self.root / "test" / "leak.png").write_bytes((self.root / "train" / "0" / "0.png").read_bytes())
        m = self.build()
        self.assertEqual(m, self.build())
        self.assertEqual((len(m.train), len(m.validation), len(m.test)), (3, 2, 2))
        for left, right in ((m.train, m.validation), (m.train, m.test), (m.validation, m.test)):
            self.assertFalse({e.sha256 for e in left} & {e.sha256 for e in right})
        path = self.root / "manifest.json"
        m.save(path)
        self.assertEqual(MazeManifest.load(path, self.root), m)

    def test_all_validation_duplicates_are_removed_from_training(self):
        m = self.build()
        selected = m.validation[0]
        (self.root / "train" / "0" / "duplicate.png").write_bytes((self.root / selected.path).read_bytes())
        m = self.build()
        self.assertEqual(len(m.validation), 3)
        self.assertNotIn(selected.sha256, {e.sha256 for e in m.train})
        self.assertEqual(len(m.validation_panel), 2)

    def test_leakage_insufficient_maps_and_corruption_fail(self):
        m = self.build()
        path = self.root / "manifest.json"
        m.save(path)
        with self.assertRaisesRegex(ValueError, "insufficient"):
            build_manifest(self.root)
        (self.root / "test" / "0" / "leak.png").write_bytes((self.root / "train" / "0" / "0.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "leakage"):
            self.build()
        (self.root / m.train[0].path).write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "changed"):
            MazeManifest.load(path, self.root)
