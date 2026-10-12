import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
import qmboot
import _qmb_numeric_v2 as _qmb_numeric


class ContinuumTests(unittest.TestCase):
    def test_tail_onset_and_count_match_exact(self):
        potential = qmboot.PolynomialPotential({2: '1/2'})
        bands, summary = qmboot.spectrum_to_continuum(potential, [10], cache_dir=None, progress=False)
        exact = qmboot.Bootstrap(potential, 10).run(Emax=8)
        isolated = exact[~exact.domain_truncated]
        self.assertEqual(summary.iloc[0].separated_levels, len(isolated))
        tail = exact[exact.domain_truncated].iloc[-1]
        self.assertLessEqual(abs(summary.iloc[0].tail_onset-tail.E_lower), 0.01)
        self.assertGreater(summary.iloc[0].plot_Emax, summary.iloc[0].tail_onset)

    def test_depth_with_no_isolated_levels(self):
        p = qmboot.PolynomialPotential({2: '1/2'})
        bands, summary = qmboot.spectrum_to_continuum(p, [2], cache_dir=None, progress=False)
        self.assertTrue(bands.empty)
        self.assertEqual(summary.iloc[0].separated_levels, 0)
        self.assertLess(summary.iloc[0].tail_onset, 0.01)

    def test_cache_invalidated_when_depths_change(self):
        p = qmboot.PolynomialPotential({2: '1/2'})
        with tempfile.TemporaryDirectory() as directory:
            bands, summary = qmboot.spectrum_to_continuum(p, [10], cache_dir=directory, progress=False)
            with patch.object(_qmb_numeric, 'sho_numeric_positive', side_effect=AssertionError('cache miss')):
                cached_bands, cached_summary = qmboot.spectrum_to_continuum(p, [10], cache_dir=directory, progress=False)
                pd.testing.assert_frame_equal(bands, cached_bands)
                pd.testing.assert_frame_equal(summary, cached_summary)
                with self.assertRaises(AssertionError):
                    qmboot.spectrum_to_continuum(p, [20], cache_dir=directory, progress=False)


if __name__ == '__main__':
    unittest.main()
