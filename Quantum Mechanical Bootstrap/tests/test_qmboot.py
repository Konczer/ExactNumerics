import unittest
import sympy as sp
import qmboot


class PolynomialTests(unittest.TestCase):
    def test_harmonic_moments(self):
        b = qmboot.Bootstrap(qmboot.PolynomialPotential({2: '1/2'}), 7)
        mu = b.moments('1/2')
        for n, value in enumerate(mu):
            expected = sp.factorial2(n-1)/2**(n//2) if n % 2 == 0 and n else (1 if n == 0 else 0)
            self.assertEqual(value, expected)
        self.assertTrue(b.is_positive_definite('1/2'))

    def test_general_recurrence(self):
        for coefficients in [{0: '1/3', 2: '1/2', 4: 1}, {2: 1, 4: '1/3', 6: 2}]:
            b = qmboot.Bootstrap(qmboot.PolynomialPotential(coefficients), 8, mass=2, hbar=3)
            seeds = {p: sp.Rational(p, 5) for p in b.seed_powers}
            E = sp.Rational(7, 3)
            mu = b.moments(E, seeds)
            d = b.potential.degree
            for n in range(1, len(mu)-d+1, 2):
                residual = 2*n*E*mu[n-1] - sum((2*n+j)*v*mu[n+j-1]
                    for j, v in enumerate(b.potential.coefficients))
                if n >= 3:
                    residual += b.hbar**2*n*(n-1)*(n-2)*mu[n-3]/(4*b.mass)
                self.assertEqual(residual, 0)
            matrix = b.hankel(E, seeds)
            expected = all(matrix[:i, :i].det() > 0 for i in range(1, matrix.rows+1))
            self.assertEqual(b.is_positive_definite(E, seeds), expected)

    def test_harmonic_run(self):
        import qmb_fast
        import pandas as pd
        b = qmboot.Bootstrap(qmboot.PolynomialPotential({2: '1/2'}), 4)
        pd.testing.assert_frame_equal(b.run(root_digits=12), qmb_fast.bootstrap_dataframe(4, root_digits=12))

    def test_seed_validation_and_scan(self):
        b = qmboot.Bootstrap(qmboot.PolynomialPotential({2: '1/2', 4: 1}), 4)
        with self.assertRaises(ValueError):
            b.moments(1)
        with self.assertRaises(NotImplementedError):
            b.run()
        self.assertEqual(len(b.scan([1, 2], {2: ['1/4', '1/2']})), 4)

    def test_invalid_potential(self):
        for coefficients in [{1: 1, 4: 1}, {4: -1}, {0: 1}]:
            with self.assertRaises(ValueError):
                qmboot.PolynomialPotential(coefficients)


if __name__ == '__main__':
    unittest.main()
