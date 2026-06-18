from pathlib import Path
import tempfile
import unittest

from coastal_segformer_demo.config import load_config
from coastal_segformer_demo.demo import run_demo


class TestDemo(unittest.TestCase):
    def test_run_demo(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            configs = root / "configs"
            configs.mkdir()
            path = configs / "demo.yml"
            path.write_text("output_dir: outputs/demo\nwidth: 64\nheight: 64\ntile_size: 32\nstride: 32\n", encoding="utf-8")
            result = run_demo(load_config(path))
            names = {p.name for p in result.artifacts}
            self.assertIn("segmentation_triplet.png", names)
            self.assertIn("class_summary.csv", names)
            for artifact in result.artifacts:
                self.assertTrue(artifact.exists(), artifact)


if __name__ == "__main__":
    unittest.main()

