"""Check that a finer driver preserves existing racing lines when resampled."""

import unittest

import numpy as np

from race import Driver


class DriverResolutionTest(unittest.TestCase):
    def test_existing_profiles_keep_their_commands_at_finer_resolution(self):
        for sectors in (6, 12):
            driver = Driver(
                np.linspace(0.2, 0.7, sectors).tolist(),
                np.linspace(-0.1, 0.2, sectors).tolist(),
                0.6,
                1.2,
            )
            fine = driver.resample(24)
            for progress in np.linspace(0, 1, 201):
                for name in ("speed", "offset"):
                    self.assertAlmostEqual(
                        driver.interpolate(getattr(driver, name), progress, 1.0),
                        fine.interpolate(getattr(fine, name), progress, 1.0),
                        places=12,
                    )
            self.assertEqual(fine.vector().shape, (50,))
            np.testing.assert_array_equal(
                Driver.from_vector(fine.vector()).vector(), fine.vector()
            )


if __name__ == "__main__":
    unittest.main()
