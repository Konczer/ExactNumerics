import unittest
import pandas as pd
import qmboot


class EnergyTableTests(unittest.TestCase):
    def test_small_offsets_survive_float_endpoint_rounding(self):
        from decimal import Decimal
        bands = pd.DataFrame([dict(K=200, level=0, E_lower=.5, E_upper=.5,
            left_clipped=False, right_clipped=False, left_offset='1e-25',
            right_offset='3e-25', left_width_upper=2e-25, right_width_upper=4e-25)])
        row = qmboot.energy_table(bands, qmboot.PolynomialPotential({2: '1/2'})).iloc[0]
        self.assertEqual(row.epsilon_K, Decimal('3e-25'))
        self.assertEqual(row.E_max-row.E_min, Decimal('4e-25'))
        self.assertEqual(row['Width upper bound'], Decimal('6e-25'))

    def test_normalization_selection_and_endpoint_error(self):
        bands = pd.DataFrame([
            dict(K=10, level=0, E_lower=.8, E_upper=1.4, left_clipped=False, right_clipped=False),
            dict(K=20, level=0, E_lower=.9, E_upper=1.3, left_clipped=False, right_clipped=False),
            dict(K=20, level=1, E_lower=2., E_upper=4., left_clipped=True, right_clipped=False),
        ])
        p = qmboot.PolynomialPotential({2: 2})  # omega = 2
        table = qmboot.energy_table(bands, p)
        self.assertEqual(list(table.columns), ['Energy Level', 'E_min', 'E_max', 'E_Theoretical', 'Width upper bound', 'epsilon_K'])
        self.assertEqual(len(table), 1)
        row = table.iloc[0]
        self.assertAlmostEqual(float(row.E_min), .45)
        self.assertAlmostEqual(float(row.E_max), .65)
        self.assertAlmostEqual(float(row.E_Theoretical), .5)
        self.assertAlmostEqual(float(row.epsilon_K), .15)
        selected = qmboot.energy_table(bands, p, K=10)
        self.assertAlmostEqual(float(selected.iloc[0].E_min), .4)


if __name__ == '__main__':
    unittest.main()
