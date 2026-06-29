import unittest

import numpy as np

from coastal_segformer_inference.segformer_adapter import prepare_rgb_for_segformer, validate_logits_shape


class TestSegFormerAdapter(unittest.TestCase):
    def test_prepare_rgb_for_segformer(self):
        bands = np.zeros((4, 8, 6), dtype="float32")
        bands[0] = 0.2
        bands[1] = 0.5
        bands[2] = 1.2
        rgb = prepare_rgb_for_segformer(bands)
        self.assertEqual(rgb.shape, (8, 6, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(int(rgb[..., 2].max()), 255)

    def test_validate_logits_shape_rejects_wrong_class_count(self):
        logits = np.zeros((1, 5, 8, 6), dtype="float32")
        with self.assertRaises(ValueError):
            validate_logits_shape(logits, expected_classes=3, height=8, width=6)

    def test_validate_logits_shape_accepts_expected_contract(self):
        logits = np.zeros((1, 3, 8, 6), dtype="float32")
        out = validate_logits_shape(logits, expected_classes=3, height=8, width=6)
        self.assertEqual(out.shape, (3, 8, 6))


if __name__ == "__main__":
    unittest.main()
