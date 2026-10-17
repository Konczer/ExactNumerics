import sympy as sp
import pandas as pd
import matplotlib.pyplot as plt

from dataclasses import dataclass
from functools import lru_cache


# ============================================================
# Quantum Harmonic Oscillator Bootstrap
#
# Dimensionless variables:
#
#     q = sqrt(m*omega/hbar) x
#     epsilon = E/(hbar*omega)
#
# Recurrence:
#
#     s <q^s>
#       = 2(s-1) epsilon <q^(s-2)>
#       + [(s-1)(s-2)(s-3)/4] <q^(s-4)>
#
# Odd moments vanish.
# ============================================================


epsilon = sp.Symbol("epsilon", real=True)


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class RootBracket:
    lo: sp.Rational
    hi: sp.Rational
    multiplicity: int = 1

    @property
    def midpoint(self):
        return (self.lo + self.hi) / 2

    @property
    def width(self):
        return self.hi - self.lo


@dataclass(frozen=True)
class Boundary:
    kind: str                  # "domain" or "root"
    exact: sp.Rational = None
    root: RootBracket = None

    @property
    def estimate(self):
        if self.kind == "domain":
            return self.exact
        return self.root.midpoint

    @property
    def uncertainty(self):
        if self.kind == "domain":
            return sp.Rational(0)
        return self.root.width / 2


@dataclass(frozen=True)
class AllowedInterval:
    K: int
    left: Boundary
    right: Boundary

    @property
    def midpoint(self):
        return (self.left.estimate + self.right.estimate) / 2

    @property
    def half_width(self):
        return (self.right.estimate - self.left.estimate) / 2

    @property
    def touches_domain(self):
        return (
            self.left.kind == "domain"
            or self.right.kind == "domain"
        )


# ============================================================
# Utilities
# ============================================================

def Q(value):
    """
    Convert a number to an exact SymPy Rational.

    Using the string representation avoids turning
    binary floating-point error into the rational.
    """
    if isinstance(value, sp.Rational):
        return value

    return sp.Rational(str(value))


# ============================================================
# Exact SHO moments
# ============================================================

@lru_cache(maxsize=None)
def sho_moments(K):
    """
    Return exact moments

        mu_s = <q^s>

    for s = 0,...,2K-2.

    A K x K Hankel matrix requires moments through
    order 2K-2.
    """

    if K < 1:
        raise ValueError("K must be >= 1")

    max_power = 2 * K - 2

    mu = [sp.Integer(0)] * (max_power + 1)

    # Normalisation
    mu[0] = sp.Integer(1)

    for s in range(1, max_power + 1):

        # Parity of SHO eigenstates
        if s % 2 == 1:
            mu[s] = sp.Integer(0)
            continue

        # Recurrence
        term = (
            2
            * (s - 1)
            * epsilon
            * mu[s - 2]
        )

        if s >= 4:
            term += (
                sp.Rational(
                    (s - 1) * (s - 2) * (s - 3),
                    4
                )
                * mu[s - 4]
            )

        mu[s] = sp.cancel(term / s)

    return tuple(mu)


# ============================================================
# Hankel / moment matrices
# ============================================================

@lru_cache(maxsize=None)
def parity_blocks(K):
    """
    Construct the even and odd parity Hankel blocks.

    Original basis:

        {1, q, q^2, ..., q^(K-1)}

    Since odd moments vanish, the full Hankel matrix
    decomposes into even and odd sectors.
    """

    mu = sho_moments(K)

    even_powers = list(range(0, K, 2))
    odd_powers  = list(range(1, K, 2))

    M_even = sp.Matrix([
        [mu[i + j] for j in even_powers]
        for i in even_powers
    ])

    M_odd = sp.Matrix([
        [mu[i + j] for j in odd_powers]
        for i in odd_powers
    ])

    return M_even, M_odd


def full_moment_matrix(K):
    """
    Construct the complete K x K moment matrix

        M_ij = <q^(i+j)>.
    """

    mu = sho_moments(K)

    return sp.Matrix([
        [mu[i + j] for j in range(K)]
        for i in range(K)
    ])



# ============================================================
# Exact determinant polynomials
# ============================================================

@lru_cache(maxsize=None)
def determinant_polynomials(K):
    """
    Return the exact determinant polynomials of the
    even and odd parity blocks.
    """

    M_even, M_odd = parity_blocks(K)

    polynomials = []

    for M in (M_even, M_odd):

        if M.rows == 0:
            polynomials.append(
                sp.Poly(1, epsilon, domain=sp.QQ)
            )
            continue

        det = sp.cancel(
            M.det(method="domain-ge")
        )

        numerator, denominator = sp.fraction(det)

        # For SHO the denominator should be a rational
        # constant and therefore irrelevant to zero locations.
        if epsilon in denominator.free_symbols:
            raise RuntimeError(
                "Unexpected epsilon-dependent denominator."
            )

        poly = sp.Poly(
            sp.expand(numerator),
            epsilon,
            domain=sp.QQ
        )

        polynomials.append(poly)

    return polynomials[0], polynomials[1]


@lru_cache(maxsize=None)
def singularity_polynomial(K):
    """
    Product

        det(M_even) det(M_odd).

    Every possible change in matrix inertia must occur
    at a zero of this polynomial.
    """

    P_even, P_odd = determinant_polynomials(K)

    return sp.Poly(
        P_even.as_expr() * P_odd.as_expr(),
        epsilon,
        domain=sp.QQ
    )


# ============================================================
# Exact positive-definiteness tests
# ============================================================

def block_is_positive_definite_exact(M, E):
    """
    Exact Sylvester criterion.

    At a rational energy away from determinant roots,
    the matrix is positive definite iff every leading
    principal minor is strictly positive.
    """

    if M.rows == 0:
        return True

    E = Q(E)

    M_E = M.subs(epsilon, E)

    for n in range(1, M_E.rows + 1):

        minor = M_E[:n, :n]

        determinant = sp.cancel(
            minor.det(method="domain-ge")
        )

        if determinant <= 0:
            return False

    return True


def is_positive_definite_exact(K, E):
    """
    Test both parity sectors exactly.
    """

    M_even, M_odd = parity_blocks(K)

    return (
        block_is_positive_definite_exact(M_even, E)
        and
        block_is_positive_definite_exact(M_odd, E)
    )


# ============================================================
# Exact root isolation
# ============================================================

def isolate_singular_roots(
    K,
    Emin=0,
    Emax=6,
    digits=30
):
    """
    Isolate every real root of

        det(M_even) det(M_odd)

    inside the requested energy domain.

    SymPy returns exact rational brackets around each root.
    """

    Emin = Q(Emin)
    Emax = Q(Emax)

    if Emin >= Emax:
        raise ValueError("Require Emin < Emax")

    P = singularity_polynomial(K)

    # Target width for root-isolation brackets
    tolerance = sp.Rational(
        1,
        10**digits
    )

    root_intervals = sp.polys.polytools.intervals(
        P,
        eps=tolerance
    )

    roots = []

    for interval, multiplicity in root_intervals:

        lo, hi = interval

        lo = sp.Rational(lo)
        hi = sp.Rational(hi)

        # Discard roots outside search region
        if hi <= Emin:
            continue

        if lo >= Emax:
            continue

        roots.append(
            RootBracket(
                lo=lo,
                hi=hi,
                multiplicity=int(multiplicity)
            )
        )

    roots.sort(
        key=lambda r: r.midpoint
    )

    return roots


import time
# ============================================================
# Bootstrap allowed regions
# ============================================================

def allowed_intervals(
    K,
    Emin=0,
    Emax=6,
    root_digits=30
):
    """
    Determine every finite-K bootstrap-allowed energy region.

    No energy grid is used.

    Algorithm
    ---------
    1. Find every determinant zero.
    2. These roots partition the energy axis into cells.
    3. Pick one exact rational point in every cell.
    4. Test moment-matrix positivity exactly.
    5. Merge adjacent allowed cells.

    Since matrix inertia cannot change without an eigenvalue
    crossing zero, one positivity test per cell is sufficient.
    """

    Emin = Q(Emin)
    Emax = Q(Emax)

    roots = isolate_singular_roots(
        K,
        Emin,
        Emax,
        digits=root_digits
    )

    # --------------------------------------------------------
    # Classify every open cell
    # --------------------------------------------------------

    allowed = []

    number_of_cells = len(roots) + 1

    for j in range(number_of_cells):

        if j == 0:
            left = Emin
        else:
            left = roots[j - 1].hi

        if j == len(roots):
            right = Emax
        else:
            right = roots[j].lo

        if left >= right:
            allowed.append(False)
            continue

        # Exact rational test point
        sample = (left + right) / 2

        allowed.append(
            is_positive_definite_exact(K, sample)
        )

    # --------------------------------------------------------
    # Convert positive cells into connected intervals
    # --------------------------------------------------------

    regions = []

    j = 0

    while j < len(allowed):

        if not allowed[j]:
            j += 1
            continue

        start = j

        while (
            j + 1 < len(allowed)
            and allowed[j + 1]
        ):
            j += 1

        end = j

        # Left boundary
        if start == 0:

            left_boundary = Boundary(
                kind="domain",
                exact=Emin
            )

        else:

            left_boundary = Boundary(
                kind="root",
                root=roots[start - 1]
            )

        # Right boundary
        if end == len(roots):

            right_boundary = Boundary(
                kind="domain",
                exact=Emax
            )

        else:

            right_boundary = Boundary(
                kind="root",
                root=roots[end]
            )

        regions.append(
            AllowedInterval(
                K=K,
                left=left_boundary,
                right=right_boundary
            )
        )

        j += 1

    return regions


# ============================================================
# Convert results to Pandas
# ============================================================

def bootstrap_dataframe(
    K,
    Emin=0,
    Emax=6,
    root_digits=30,
    decimal_digits=20
):
    """
    Run the bootstrap and return a tidy Pandas DataFrame.
    """

    regions = allowed_intervals(
        K,
        Emin,
        Emax,
        root_digits
    )

    rows = []

    for level, region in enumerate(regions):

        # Ignore intervals which simply continue to the
        # artificial boundary of our search domain when
        # identifying resolved states.
        rows.append({
            "K": K,
            "level": level,

            "E_lower":
                float(sp.N(
                    region.left.estimate,
                    decimal_digits
                )),

            "E_upper":
                float(sp.N(
                    region.right.estimate,
                    decimal_digits
                )),

            "E_mid":
                float(sp.N(
                    region.midpoint,
                    decimal_digits
                )),

            "half_width":
                float(sp.N(
                    region.half_width,
                    decimal_digits
                )),

            "left_root_uncertainty":
                float(sp.N(
                    region.left.uncertainty,
                    decimal_digits
                )),

            "right_root_uncertainty":
                float(sp.N(
                    region.right.uncertainty,
                    decimal_digits
                )),

            "domain_truncated":
                region.touches_domain
        })

    return pd.DataFrame(rows)


# ============================================================
# Multiple bootstrap depths
# ============================================================

def bootstrap_sequence(
    Ks,
    Emin=0,
    Emax=6,
    root_digits=30
):
    """
    Run several bootstrap truncation orders
    and report computation time for each depth.
    """

    frames = []

    for K in Ks:

        print(f"\nRunning K = {K} ...")

        start = time.perf_counter()

        df = bootstrap_dataframe(
            K,
            Emin,
            Emax,
            root_digits
        )

        elapsed = time.perf_counter() - start

        # Add runtime to results
        df["runtime_seconds"] = elapsed

        frames.append(df)

        print(
            f"K = {K} completed in "
            f"{elapsed:.2f} seconds"
        )

    return pd.concat(
        frames,
        ignore_index=True
    )

# ============================================================
# Plot bootstrap spectrum
# ============================================================

def plot_bootstrap_regions(results):
    """
    Plot allowed energy intervals against bootstrap depth.
    """

    plt.figure(figsize=(10, 6))

    for _, row in results.iterrows():

        plt.plot(
            [row["E_lower"], row["E_upper"]],
            [row["K"], row["K"]],
            linewidth=4
        )

    plt.xlabel(
        r"Dimensionless energy $\epsilon = E/(\hbar\omega)$"
    )

    plt.ylabel(
        r"Bootstrap depth $K$"
    )

    plt.title(
        "Quantum Harmonic Oscillator Moment Bootstrap"
    )

    plt.grid(alpha=0.25)

    plt.show()
# Override the symbolic hotspots with compiled exact kernels.
import operator
import _qmb_native

def _depth(K):
    K = operator.index(K)
    if K < 1:
        raise ValueError("K must be >= 1")
    return K

@lru_cache(maxsize=None)
def determinant_polynomials(K):
    coefficients = _qmb_native.determinant_coefficients(_depth(K))
    return tuple(
        sp.Poly.from_list([sp.Rational(c) for c in reversed(block)],
                          epsilon, domain=sp.QQ)
        for block in coefficients
    )

def is_positive_definite_exact(K, E):
    E = Q(E)
    return _qmb_native.is_positive_definite(_depth(K), str(E.p), str(E.q))
