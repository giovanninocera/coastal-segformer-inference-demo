import unittest

import numpy as np

from coastal_segformer_inference.synthetic import _coastal_zones


class TestSyntheticGeometry(unittest.TestCase):
    def test_lagoon_connects_to_open_sea(self):
        water, lagoon = _coastal_zones(256, 160)
        open_sea = water & ~lagoon
        neighboring_sea = (
            np.roll(open_sea, 1, axis=0)
            | np.roll(open_sea, -1, axis=0)
            | np.roll(open_sea, 1, axis=1)
            | np.roll(open_sea, -1, axis=1)
        )
        self.assertTrue(np.any(lagoon & neighboring_sea))
        self.assertGreater(float(open_sea[0].mean()), 0.9)


if __name__ == "__main__":
    unittest.main()
