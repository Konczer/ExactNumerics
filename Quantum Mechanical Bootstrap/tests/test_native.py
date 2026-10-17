"""Run with: python -m unittest discover -s tests -v"""
import unittest
import sympy as sp
import qmb_fast as fast


class ExactKernelTests(unittest.TestCase):
    def test_determinants(self):
        for K in range(1, 9):
            for actual, matrix in zip(fast.determinant_polynomials(K), fast.parity_blocks(K)):
                expected = matrix.det(method="domain-ge") if matrix.rows else sp.Integer(1)
                self.assertEqual(sp.cancel(actual.as_expr() - expected), 0, K)

    def test_positivity(self):
        for K in range(1, 9):
            for E in [sp.Rational(-1), sp.Rational(0), sp.Rational(1, 2),
                      sp.Rational(3, 2), sp.Rational(2, 3), sp.Rational(7, 3),
                      sp.Rational(11, 2), sp.Rational(6)]:
                expected = all(fast.block_is_positive_definite_exact(m, E)
                               for m in fast.parity_blocks(K))
                self.assertEqual(fast.is_positive_definite_exact(K, E), expected, (K, E))

    def test_intervals(self):
        original = fast.is_positive_definite_exact
        for K in [2, 4, 6]:
            native = fast.allowed_intervals(K, root_digits=12)
            try:
                fast.is_positive_definite_exact = lambda k, e: all(
                    fast.block_is_positive_definite_exact(m, e) for m in fast.parity_blocks(k))
                reference = fast.allowed_intervals(K, root_digits=12)
            finally:
                fast.is_positive_definite_exact = original
            self.assertEqual(native, reference)

    def test_invalid_depth(self):
        with self.assertRaises(ValueError):
            fast.is_positive_definite_exact(0, 1)


if __name__ == "__main__":
    unittest.main()
