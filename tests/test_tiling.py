import unittest

import numpy as np

from coastal_segformer_inference.tiling import crop, make_windows, stitch_logits


class TestTiling(unittest.TestCase):
    def test_windows_cover_shape(self):
        windows = make_windows(100, 120, 32, 32)
        self.assertTrue(windows)
        self.assertEqual(windows[-1].row + windows[-1].height, 100)
        self.assertEqual(windows[-1].col + windows[-1].width, 120)

    def test_stitch_logits(self):
        arr = np.ones((2, 32, 32), dtype="float32")
        windows = make_windows(32, 32, 16, 16)
        tiles = [crop(arr, w) for w in windows]
        stitched = stitch_logits(tiles, windows, 32, 32)
        self.assertEqual(stitched.shape, (2, 32, 32))
        self.assertAlmostEqual(float(stitched.mean()), 1.0)


if __name__ == "__main__":
    unittest.main()
