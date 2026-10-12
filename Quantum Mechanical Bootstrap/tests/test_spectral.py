import unittest
import sympy as sp
import pandas as pd
import qmboot
import _qmb_numeric_v2 as _qmb_numeric
from pathlib import Path
import tempfile
from unittest.mock import patch


class SpectralTests(unittest.TestCase):
    def test_adaptive_refines_resolved_widths(self):
        p = qmboot.PolynomialPotential({2: '1/2'})
        bands = qmboot.spectral_sequence(p, [20], Emax=1, boundary_digits=8,
            relative_width_tolerance=1e-5, progress=False)
        row = bands.iloc[0]
        self.assertTrue(row.relative_width_resolved)
        width = row.left_width_upper+row.right_width_upper
        self.assertLessEqual(2*row.boundary_tolerance, 1e-5*width)
        exact = qmboot.spectral_sequence(p, [20], Emax=1, boundary_digits=14,
            progress=False).iloc[0]
        self.assertAlmostEqual(row.E_lower, exact.E_lower, places=11)
        self.assertAlmostEqual(row.E_upper, exact.E_upper, places=11)

    def test_adaptive_keeps_unresolved_width_as_upper_limit(self):
        p = qmboot.PolynomialPotential({2: '1/2'})
        bands = qmboot.spectral_sequence(p, [100], Emax=1, boundary_digits=12,
            relative_width_tolerance=1e-5, progress=False)
        row = bands.iloc[0]
        self.assertFalse(row.relative_width_resolved)
        self.assertTrue(row.left_unresolved and row.right_unresolved)
        self.assertLessEqual(row.left_width_upper+row.right_width_upper, 2e-12)

    def test_uncapped_depth(self):
        # Negative energy exits before LDL but exercises allocation/recurrence
        # above the previous cap in the actual C++ backend.
        self.assertEqual(_qmb_numeric.sho_numeric_positive(500, '-1', 160), -1)
        p = qmboot.PolynomialPotential({2: '1/2'})
        with patch.object(_qmb_numeric, 'sho_numeric_positive', return_value=1):
            bands = qmboot.spectral_sequence(p, [500], Emax=1, progress=False)
        self.assertEqual(list(bands.K), [500])

    def test_positivity_matches_exact(self):
        for K in [4, 10, 16]:
            b = qmboot.Bootstrap(qmboot.PolynomialPotential({2: '1/2'}), K)
            for energy in ['0.49', '0.5', '0.51', '1.5', '2.1', '5.5']:
                status = _qmb_numeric.sho_numeric_positive(K, energy, 160)
                self.assertNotEqual(status, 0)
                self.assertEqual(status > 0, b.is_positive_definite(energy))

    def test_boundaries_match_exact(self):
        b = qmboot.Bootstrap(qmboot.PolynomialPotential({2: '1/2'}), 10)
        exact = b.run(root_digits=18)
        numerical = qmboot.spectral_sequence(b.potential, [10], boundary_digits=12, progress=False)
        for _, row in numerical.iterrows():
            containing = exact[(exact.E_lower < row.E_level) & (exact.E_upper > row.E_level)]
            self.assertEqual(len(containing), 1)
            band = containing.iloc[0]
            if not row.left_clipped:
                self.assertAlmostEqual(row.E_lower, band.E_lower, places=10)
            if not row.right_clipped:
                self.assertAlmostEqual(row.E_upper, band.E_upper, places=10)

    def test_high_depth_and_resolution(self):
        p = qmboot.PolynomialPotential({2: '1/2'})
        result = qmboot.spectral_sequence(p, [50, 100, 200], Emax=1, boundary_digits=12, progress=False)
        self.assertEqual(list(result.K), [50, 100, 200])
        self.assertTrue((result.left_width_upper <= 1e-12).all())
        self.assertTrue((result.right_width_upper <= 1e-12).all())
        for K in [100, 200]:
            self.assertEqual(_qmb_numeric.sho_numeric_positive(K, '0.5', 240), 1)

    def test_potential_validation(self):
        with self.assertRaises(ValueError):
            qmboot.spectral_sequence(qmboot.PolynomialPotential({4: 1}), [10], progress=False)

    def test_cache_and_parameter_changes(self):
        p = qmboot.PolynomialPotential({2: '1/2'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bands.csv'
            original = qmboot.spectral_sequence(p, [10], Emax=1, boundary_digits=6,
                                               progress=False, cache_path=path)
            with patch.object(_qmb_numeric, 'sho_numeric_positive', side_effect=AssertionError('cache missed')):
                cached = qmboot.spectral_sequence(p, [10], Emax=1, boundary_digits=6,
                                                 progress=False, cache_path=path)
                pd.testing.assert_frame_equal(original, cached, check_exact=False)
                with self.assertRaises(AssertionError):
                    qmboot.spectral_sequence(p, [20], Emax=1, boundary_digits=6,
                                             progress=False, cache_path=path)

    def test_incomplete_cache_recomputed(self):
        p = qmboot.PolynomialPotential({2: '1/2'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bands.csv'
            expected = qmboot.spectral_sequence(p, [10], Emax=1, boundary_digits=6,
                                               progress=False, cache_path=path)
            path.write_text('K,level\n10,0\n')
            actual = qmboot.spectral_sequence(p, [10], Emax=1, boundary_digits=6,
                                             progress=False, cache_path=path)
            pd.testing.assert_frame_equal(expected.drop(columns='runtime_seconds'),
                                          actual.drop(columns='runtime_seconds'))

    def test_auto_uses_numerical_for_large_depth(self):
        b = qmboot.Bootstrap(qmboot.PolynomialPotential({2: '1/2'}), 100, mass=2, hbar=3)
        with patch('qmboot.spectral_sequence', return_value='numerical') as numerical:
            self.assertEqual(b.run(method='auto'), 'numerical')
            self.assertEqual(numerical.call_args.kwargs['Ks'], [100])
            self.assertEqual(numerical.call_args.kwargs['mass'], 2)
            self.assertEqual(numerical.call_args.kwargs['hbar'], 3)

    def test_numerical_energy_scaling(self):
        p = qmboot.PolynomialPotential({2: '1/2'})
        result = qmboot.spectral_sequence(p, [10], Emax=3, mass=4, hbar=2,
                                          boundary_digits=6, progress=False)
        self.assertEqual(list(result.E_level), [0.5, 1.5, 2.5])


if __name__ == '__main__':
    unittest.main()
