from pathlib import Path
import tempfile
import unittest

from coastal_segformer_inference.config import load_config


class TestConfig(unittest.TestCase):
    def test_rejects_segformer_backend_without_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference.yml"
            path.write_text("model_backend: segformer\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(path)

    def test_loads_heuristic_backend_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inference.yml"
            path.write_text("output_dir: outputs/test\n", encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.model_backend, "heuristic")
            self.assertIsNone(config.checkpoint)


if __name__ == "__main__":
    unittest.main()
