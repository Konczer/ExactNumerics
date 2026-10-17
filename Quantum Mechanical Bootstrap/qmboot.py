"""Exact moment bootstrap for even, confining polynomial potentials.

Convention: H = -hbar**2/(2*mass) d**2/dx**2 + V(x).
"""
from dataclasses import dataclass
from itertools import product
from collections.abc import Mapping
import operator
import sympy as sp


def _rational(value):
    return sp.Rational(str(value))


def _native():
    try:
        import _qmb_native
    except ImportError as exc:
        raise ImportError("Build qmboot first: python -m pip install . (see README.md)") from exc
    return _qmb_native


@dataclass(frozen=True, init=False)
class PolynomialPotential:
    """Specify V(x) as {power: coefficient}, e.g. {2: 0.5, 4: 1}."""
    coefficients: tuple

    def __init__(self, coefficients):
        if not isinstance(coefficients, Mapping) or not coefficients:
            raise ValueError("Provide a nonempty mapping {power: coefficient}")
        terms = {}
        for power, coefficient in coefficients.items():
            power = operator.index(power)
            if power < 0:
                raise ValueError("Polynomial powers must be nonnegative")
            coefficient = _rational(coefficient)
            if coefficient:
                terms[power] = coefficient
        degree = max(terms, default=0)
        if degree < 2 or any(power % 2 for power in terms):
            raise ValueError("Currently supported: even potentials of degree >= 2")
        if terms[degree] <= 0:
            raise ValueError("Leading coefficient must be positive (confining potential)")
        object.__setattr__(self, "coefficients", tuple(terms.get(i, sp.S.Zero) for i in range(degree+1)))

    @property
    def degree(self):
        return len(self.coefficients)-1

    def as_expr(self, x=None):
        x = sp.Symbol("x") if x is None else x
        return sum(c*x**i for i, c in enumerate(self.coefficients))


@dataclass(frozen=True)
class Bootstrap:
    potential: PolynomialPotential
    hankel_depth: int = 10
    mass: object = 1
    hbar: object = 1

    def __post_init__(self):
        if not isinstance(self.potential, PolynomialPotential):
            raise TypeError("potential must be a PolynomialPotential")
        depth = operator.index(self.hankel_depth)
        if depth < 1:
            raise ValueError("hankel_depth must be >= 1")
        object.__setattr__(self, "hankel_depth", depth)
        for name in ("mass", "hbar"):
            value = _rational(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

    @property
    def seed_powers(self):
        """Independent moments required in addition to energy."""
        return tuple(range(2, self.potential.degree, 2))

    def _arguments(self, energy, seed_moments):
        seeds = {} if seed_moments is None else seed_moments
        if set(seeds) != set(self.seed_powers):
            raise ValueError(f"Provide exactly these seed moment powers: {self.seed_powers}")
        return (self.hankel_depth, [str(c) for c in self.potential.coefficients],
                str(_rational(energy)), [str(_rational(seeds[p])) for p in self.seed_powers],
                str(self.mass), str(self.hbar))

    def moments(self, energy, seed_moments=None):
        """Exact moments mu_0,...,mu_(2K-2), computed in C++."""
        return tuple(sp.Rational(c) for c in _native().polynomial_moments(
            *self._arguments(energy, seed_moments)))

    def hankel(self, energy, seed_moments=None):
        mu = self.moments(energy, seed_moments)
        return sp.Matrix(self.hankel_depth, self.hankel_depth, lambda i, j: mu[i+j])

    def is_positive_definite(self, energy, seed_moments=None):
        return _native().polynomial_positive(*self._arguments(energy, seed_moments))

    def scan(self, energies, seed_grids=None):
        """Classify explicit samples; this is not certified interval isolation.

        Quartic example: scan(energies, seed_grids={2: x2_values}).
        """
        import pandas as pd
        grids = {} if seed_grids is None else seed_grids
        if set(grids) != set(self.seed_powers):
            raise ValueError(f"Provide seed grids for powers {self.seed_powers}")
        combinations = list(product(*(grids[p] for p in self.seed_powers)))
        rows = []
        for energy in energies:
            for values in combinations:
                seeds = dict(zip(self.seed_powers, values))
                rows.append({"K": self.hankel_depth, "energy": float(_rational(energy)),
                             **{f"mu_{p}": float(_rational(v)) for p, v in seeds.items()},
                             "positive_definite": self.is_positive_definite(energy, seeds)})
        return pd.DataFrame(rows, columns=["K", "energy", *[f"mu_{p}" for p in self.seed_powers], "positive_definite"])

    def run(self, Emin=0, Emax=6, root_digits=30, method="exact", cache_path=None):
        """Certified energy brackets for quadratic potentials.

        method='auto' selects numerical level-centred bands above K=12;
        those results are not certified. method='exact' is the default.
        For degree >= 4, scan energy and independent seed moments instead.
        """
        if self.potential.degree != 2:
            raise NotImplementedError("Higher-degree potentials need seed moment search; use scan().")
        if method not in ("exact", "numerical", "auto"):
            raise ValueError("method must be exact, numerical, or auto")
        if method == "numerical" or (method == "auto" and self.hankel_depth > 12):
            return spectral_sequence(self.potential, Ks=[self.hankel_depth], Emin=Emin, Emax=Emax,
                                     boundary_digits=min(operator.index(root_digits), 80),
                                     mass=self.mass, hbar=self.hbar, cache_path=cache_path)
        import qmb_fast
        shift = self.potential.coefficients[0]
        scale = self.hbar*sp.sqrt(2*self.potential.coefficients[2]/self.mass)
        # Keep rational search bounds so the underlying root isolation stays exact.
        if not scale.is_Rational:
            raise ValueError("Certified run currently requires rational hbar*sqrt(2*v2/mass); use scan().")
        result = qmb_fast.bootstrap_dataframe(self.hankel_depth,
            (_rational(Emin)-shift)/scale, (_rational(Emax)-shift)/scale, root_digits)
        for column in ["E_lower", "E_upper", "E_mid"]:
            if column in result:
                result[column] = result[column]*float(scale)+float(shift)
        for column in ["half_width", "left_root_uncertainty", "right_root_uncertainty"]:
            if column in result:
                result[column] *= float(scale)
        return result


def plot_regions(results):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in results.iterrows():
        ax.plot([row["E_lower"], row["E_upper"]], [row["K"], row["K"]], linewidth=4)
    ax.set(xlabel="Energy E", ylabel="Hankel depth K", title="Moment bootstrap allowed energy regions")
    ax.grid(alpha=0.25)
    plt.show()
    return ax


def spectral_sequence(potential, Ks=(10, 20, 50, 100, 200), Emin=0, Emax=6,
                      boundary_digits=24, precision=160, verify_precision=True,
                      progress=True, cache_path=None, mass=1, hbar=1,
                      relative_width_tolerance=None, max_boundary_digits=80):
    """Numerical SHO bands near each known oscillator level, for large K.

    Uses scaled high-precision LDL, not symbolic determinant roots. Each level
    is searched inside its nearest-level window. Window clipping is reported;
    this is a level-centred continuation search, not certified root isolation.
    Unresolved widths are reported as upper bounds at 10**(-boundary_digits).
    """
    from decimal import Decimal, localcontext
    import time
    import pandas as pd
    if not isinstance(potential, PolynomialPotential) or potential.degree != 2:
        raise ValueError("spectral_sequence currently supports quadratic potentials")
    if precision not in (160, 240):
        raise ValueError("precision must be 160 or 240")
    boundary_digits = operator.index(boundary_digits)
    if not 2 <= boundary_digits <= 80:
        raise ValueError("boundary_digits must be 2..80")
    if relative_width_tolerance is not None:
        relative_width_tolerance = sp.Rational(str(relative_width_tolerance))
        max_boundary_digits = operator.index(max_boundary_digits)
        if not 0 < relative_width_tolerance < 1:
            raise ValueError('relative_width_tolerance must be between 0 and 1')
        if not boundary_digits <= max_boundary_digits <= 80:
            raise ValueError('max_boundary_digits must be between boundary_digits and 80')
    depths = tuple(operator.index(k) for k in Ks)
    if not depths or any(k < 2 for k in depths) or len(set(depths)) != len(depths):
        raise ValueError("Ks must contain unique integer depths >= 2")
    mass, hbar = _rational(mass), _rational(hbar)
    if mass <= 0 or hbar <= 0:
        raise ValueError("mass and hbar must be positive")
    shift = potential.coefficients[0]
    scale = hbar*sp.sqrt(2*potential.coefficients[2]/mass)
    lo, hi = _rational(Emin), _rational(Emax)
    if lo >= hi:
        raise ValueError("Require Emin < Emax")
    levels = [n for n in range(max(0, int(sp.floor((lo-shift)/scale-sp.Rational(1, 2)))),
                              max(0, int(sp.ceiling((hi-shift)/scale-sp.Rational(1, 2))))+1)
              if lo <= shift+scale*(n+sp.Rational(1, 2)) <= hi]
    if not levels:
        raise ValueError("Energy domain must contain at least one oscillator level")
    import json
    from pathlib import Path
    cache_key = {
        'solver_version': 1, 'coefficients': [str(c) for c in potential.coefficients],
        'Ks': list(depths), 'Emin': str(lo), 'Emax': str(hi),
        'boundary_digits': boundary_digits, 'precision': precision,
        'verify_precision': bool(verify_precision),
    }
    if mass != 1 or hbar != 1:
        cache_key.update(mass=str(mass), hbar=str(hbar))
    if relative_width_tolerance is not None:
        cache_key.update(adaptive_version=1, relative_width_tolerance=str(relative_width_tolerance),
                         max_boundary_digits=max_boundary_digits)
    cache_file = Path(cache_path) if cache_path is not None else None
    metadata_file = cache_file.with_suffix(cache_file.suffix+'.json') if cache_file else None
    if cache_file and cache_file.exists() and metadata_file.exists():
        try:
            if json.loads(metadata_file.read_text()) == cache_key:
                cached = pd.read_csv(cache_file, dtype={
                    'left_offset': str, 'right_offset': str,
                    **{c: 'boolean' for c in ['left_unresolved', 'right_unresolved', 'left_clipped', 'right_clipped']},
                })
                required = {'K', 'level', 'E_level', 'E_lower', 'E_upper', 'left_offset',
                            'right_offset', 'left_width_upper', 'right_width_upper',
                            'left_unresolved', 'right_unresolved', 'left_clipped', 'right_clipped',
                            'boundary_tolerance', 'precision_digits', 'runtime_seconds'}
                if relative_width_tolerance is not None:
                    required.add('relative_width_resolved')
                valid = required.issubset(cached.columns) and not cached.empty
                if valid:
                    import numpy as np
                    for column in required-{'left_offset', 'right_offset', 'left_unresolved',
                                             'right_unresolved', 'left_clipped', 'right_clipped'}:
                        cached[column] = pd.to_numeric(cached[column], errors='raise')
                    numeric = cached.select_dtypes(include='number')
                    valid = (not cached[list(required)].isna().any().any()
                             and np.isfinite(numeric.to_numpy()).all()
                             and list(dict.fromkeys(cached.K)) == list(depths)
                             and set(zip(cached.K, cached.level)) == {(k, n) for k in depths for n in levels}
                             and not cached.duplicated(['K', 'level']).any()
                             and (cached.E_lower <= cached.E_level).all()
                             and (cached.E_upper >= cached.E_level).all()
                             and (cached.left_width_upper >= 0).all()
                             and (cached.right_width_upper >= 0).all())
                if valid:
                    for column in ['left_unresolved', 'right_unresolved', 'left_clipped', 'right_clipped']:
                        cached[column] = cached[column].astype(bool)
                    if progress:
                        print(f"Loaded depth comparison from {cache_file}", flush=True)
                    return cached
        except (ValueError, OSError):
            pass
    try:
        import _qmb_numeric_v2 as native
    except ImportError as exc:
        raise ImportError("Rebuild qmboot to enable large-depth comparisons; see README.md") from exc
    rows = []
    with localcontext() as ctx:
        ctx.prec = max(100, boundary_digits+30)
        D = lambda x: Decimal(str(sp.N(x, ctx.prec)))
        lower, upper = D((lo-shift)/scale), D((hi-shift)/scale)
        physical_scale, physical_shift = D(scale), D(shift)
        tolerance = Decimal(10)**(-boundary_digits)
        for K in depths:
            start = time.perf_counter()
            if progress:
                print(f"Running K = {K} (high precision numerical)...", flush=True)
            for level in levels:
                centre = Decimal(level)+Decimal('0.5')
                cache = {}

                def classify(energy, digits=precision):
                    key = (energy, digits)
                    if key not in cache:
                        cache[key] = native.sho_numeric_positive(K, str(energy), digits)
                    if cache[key] == 0:
                        if digits == 160:
                            return classify(energy, 240)
                        raise ArithmeticError(f"Unresolved LDL pivot at K={K}, E={energy}")
                    return cache[key]

                if classify(centre) != 1:
                    raise ArithmeticError(f"Lost positivity at exact oscillator level {level}, K={K}")
                if verify_precision and precision == 160 and classify(centre, 240) != 1:
                    raise ArithmeticError(f"Oscillator-level positivity changes with precision at K={K}")
                bounds = []
                for sign, edge in [(-1, max(lower, centre-Decimal('0.5'))),
                                   (1, min(upper, centre+Decimal('0.5')))]:
                    distance = abs(edge-centre)
                    if distance == 0:
                        bounds.append((Decimal(0), Decimal(0), True, False))
                        continue
                    # Trace outwards; avoids assuming the whole energy domain is monotone.
                    good, bad = Decimal(0), None
                    for step in range(1, 9):
                        offset = distance*step/8
                        if classify(centre+sign*offset) < 0:
                            bad = offset
                            break
                        good = offset
                    if bad is None:
                        if verify_precision and precision == 160 and classify(edge, 240) != 1:
                            raise ArithmeticError(f"Window-edge positivity changes with precision at K={K}")
                        bounds.append((distance, distance, True, False))
                        continue
                    while bad-good > tolerance:
                        mid = (good+bad)/2
                        if classify(centre+sign*mid) > 0:
                            good = mid
                        else:
                            bad = mid
                    if verify_precision and precision == 160:
                        # Independent arithmetic precision at both sides of each boundary.
                        if classify(centre+sign*good, 240) != 1 or classify(centre+sign*bad, 240) != -1:
                            raise ArithmeticError(f"Boundary changes with precision at K={K}; use precision=240")
                    bounds.append((good, bad, False, good == 0))
                left, right = bounds
                adaptive_resolved = False
                actual_tolerance = tolerance
                if relative_width_tolerance is not None and not (left[2] or right[2]):
                    # Refine the existing brackets, retaining all energy evaluations.
                    # A zero lower width at the initial floor is an upper limit;
                    # do not chase an arbitrarily tiny unresolved band.
                    sides = [list(left), list(right)]
                    relative = D(relative_width_tolerance)
                    floor = Decimal(10)**(-max_boundary_digits)
                    limited = False
                    while True:
                        width_lower = sides[0][0]+sides[1][0]
                        uncertainty = sum(side[1]-side[0] for side in sides)
                        if width_lower == 0:
                            break
                        if uncertainty <= relative*width_lower:
                            adaptive_resolved = True
                            break
                        target = max(floor, relative*width_lower/2)
                        changed = False
                        for sign, side in zip((-1, 1), sides):
                            while side[1]-side[0] > target:
                                mid = (side[0]+side[1])/2
                                try:
                                    verdict = classify(centre+sign*mid)
                                except ArithmeticError:
                                    limited = True
                                    break
                                side[0 if verdict > 0 else 1] = mid
                                changed = True
                            if limited:
                                break
                        if limited or not changed:
                            break
                    for side in sides:
                        side[3] = side[0] == 0
                    left, right = (tuple(side) for side in sides)
                    actual_tolerance = max(side[1]-side[0] for side in sides)
                    if verify_precision and precision == 160:
                        for sign, side in zip((-1, 1), sides):
                            if classify(centre+sign*side[0], 240) != 1 or classify(centre+sign*side[1], 240) != -1:
                                raise ArithmeticError(f'Adaptive boundary changes with precision at K={K}')
                lmid, rmid = (left[0]+left[1])/2, (right[0]+right[1])/2
                physical_centre = physical_shift+physical_scale*centre
                rows.append({
                    'K': K, 'level': level, 'E_level': float(physical_centre),
                    'E_lower': float(physical_centre-physical_scale*lmid),
                    'E_upper': float(physical_centre+physical_scale*rmid),
                    'left_offset': str(physical_scale*lmid),
                    'right_offset': str(physical_scale*rmid),
                    'left_width_upper': float(physical_scale*left[1]),
                    'right_width_upper': float(physical_scale*right[1]),
                    'left_unresolved': left[3], 'right_unresolved': right[3],
                    'left_clipped': left[2], 'right_clipped': right[2],
                    'boundary_tolerance': float(physical_scale*actual_tolerance),
                    'precision_digits': precision,
                })
                if relative_width_tolerance is not None:
                    rows[-1]['relative_width_resolved'] = adaptive_resolved
                    if progress:
                        status = 'relative tolerance met' if adaptive_resolved else 'width upper limit'
                        print(f'  level {level}: {status}', flush=True)
            elapsed = time.perf_counter()-start
            for row in rows:
                if row['K'] == K:
                    row['runtime_seconds'] = elapsed
            if progress:
                print(f"K = {K} completed in {elapsed:.2f} seconds", flush=True)
    result = pd.DataFrame(rows)
    if cache_file:
        result.to_csv(cache_file, index=False)
        metadata_file.write_text(json.dumps(cache_key, indent=2))
    return result


def plot_spectral_splitting(results, save_path=None):
    """Compare all depths and show narrowing on a logarithmic scale."""
    import matplotlib.pyplot as plt
    import numpy as np
    depths = list(dict.fromkeys(results['K']))
    colors = plt.get_cmap('viridis')(np.linspace(0.1, 0.9, len(depths)))
    fig, (spectrum, narrowing) = plt.subplots(1, 2, figsize=(14, 5), layout='constrained')
    for y, (K, color) in enumerate(zip(depths, colors)):
        group = results[results['K'] == K]
        for _, row in group.iterrows():
            spectrum.plot([row.E_lower, row.E_upper], [y, y], color=color, linewidth=5)
            spectrum.plot(row.E_level, y, '.', color=color, markersize=5)
    for energy in sorted(results.E_level.unique()):
        spectrum.axvline(energy, color='grey', alpha=0.25, linestyle=':')
    spectrum.set(yticks=range(len(depths)), yticklabels=[f'K = {k}' for k in depths],
                 xlabel='Energy E', ylabel='Hankel depth', title='Allowed bands near oscillator levels')
    for level, group in results.groupby('level'):
        group = group.sort_values('K')
        widths = group[['left_width_upper', 'right_width_upper']].max(axis=1)
        line, = narrowing.plot(group.K, widths, '.-', label=f'Level {level}')
        unresolved = ((group.left_unresolved & (group.left_width_upper >= group.right_width_upper))
                      | (group.right_unresolved & (group.right_width_upper >= group.left_width_upper)))
        narrowing.scatter(group.K[unresolved], widths[unresolved], marker='v',
                          color=line.get_color(), zorder=3)
        clipped = ((group.left_clipped & (group.left_width_upper >= group.right_width_upper))
                   | (group.right_clipped & (group.right_width_upper >= group.left_width_upper)))
        narrowing.scatter(group.K[clipped], widths[clipped], marker='s', facecolors='none',
                          edgecolors=line.get_color(), s=55, zorder=4)
    narrowing.set(yscale='log', xlabel='Hankel depth K',
                  ylabel='Distance from level within the search window',
                  title='Band narrowing (triangle: upper bound; square: window cut)')
    narrowing.legend()
    spectrum.grid(alpha=0.2)
    narrowing.grid(alpha=0.2, which='both')
    if save_path:
        fig.savefig(save_path, dpi=180)
    plt.show()
    return fig


def spectrum_to_continuum(potential, Ks=(10, 20, 50, 100, 200),
                          boundary_digits=2, tail_padding=4, cache_dir='.qmboot-cache',
                          progress=True, mass=1, hbar=1):
    """Find the sampled high-energy merged tail and all lower level bands.

    Tail identification uses three consecutive accepted integer midpoints plus
    eight samples per unit across that span, checked at 160 and 240 digits.
    This is a numerical finite-depth resolution diagnostic, not a proof of
    positivity at every larger energy. Returns (bands, summary).
    """
    import json
    import time
    from pathlib import Path
    from decimal import Decimal, localcontext
    import pandas as pd
    import _qmb_numeric_v2 as _qmb_numeric
    if not isinstance(potential, PolynomialPotential) or potential.degree != 2:
        raise ValueError('This comparison currently supports quadratic potentials')
    depths = tuple(operator.index(k) for k in Ks)
    if not depths or len(set(depths)) != len(depths) or any(k<2 for k in depths):
        raise ValueError('Choose unique integer depths >= 2')
    boundary_digits = operator.index(boundary_digits)
    if not 2 <= boundary_digits <= 80:
        raise ValueError('boundary_digits must be 2..80')
    tail_padding = operator.index(tail_padding)
    if tail_padding < 1:
        raise ValueError('tail_padding must be >= 1')
    mass, hbar = _rational(mass), _rational(hbar)
    if mass <= 0 or hbar <= 0:
        raise ValueError('mass and hbar must be positive')
    shift = potential.coefficients[0]
    scale = hbar*sp.sqrt(2*potential.coefficients[2]/mass)
    directory = Path(cache_dir) if cache_dir is not None else None
    key = {'version': 1, 'coefficients': [str(c) for c in potential.coefficients],
           'Ks': list(depths), 'boundary_digits': boundary_digits, 'tail_padding': tail_padding,
           'mass': str(mass), 'hbar': str(hbar)}
    if directory:
        directory.mkdir(parents=True, exist_ok=True)
        saved = directory/'continuum.json'
        if saved.exists():
            try:
                payload = json.loads(saved.read_text())
                if payload['parameters'] == key:
                    bands = pd.DataFrame(payload['bands'])
                    summary = pd.DataFrame(payload['summary'])
                    if (set(summary.K) == set(depths) and not summary.empty
                            and {'K','level','E_level','E_lower','E_upper'}.issubset(bands.columns)):
                        if progress:
                            print('Loaded extended spectrum from cache', flush=True)
                        return bands, summary
            except (ValueError, KeyError, OSError):
                pass
    frames, summaries = [], []
    with localcontext() as ctx:
        ctx.prec = 110
        physical_shift = Decimal(str(sp.N(shift, 110)))
        physical_scale = Decimal(str(sp.N(scale, 110)))
        tolerance = Decimal(10)**(-boundary_digits)
        for K in depths:
            start = time.perf_counter()
            if progress:
                print(f'K = {K}: finding the last separated levels...', flush=True)
            evaluations = {}

            def status(energy):
                if energy not in evaluations:
                    first = _qmb_numeric.sho_numeric_positive(K, str(energy), 160)
                    second = _qmb_numeric.sho_numeric_positive(K, str(energy), 240)
                    if second == 0 or (first != 0 and first != second):
                        raise ArithmeticError(f'Positivity is unstable at K={K}, E={energy}')
                    evaluations[energy] = second
                return evaluations[energy]

            streak, horizon = 0, None
            for integer in range(1, max(8, 2*K)+1):
                streak = streak+1 if status(Decimal(integer)) == 1 else 0
                if streak >= 3:
                    candidate = integer-2
                    if all(status(Decimal(candidate)+Decimal(j)/8)==1 for j in range(1, 17)):
                        horizon = candidate
                        break
            if horizon is None:
                raise ArithmeticError(f'No stable merged tail found for K={K}')
            # Trace backwards from an accepted midpoint to the final sampled gap.
            good, bad = Decimal(horizon), None
            for j in range(1, 17):
                probe = Decimal(horizon)-Decimal(j)/16
                if status(probe) < 0:
                    bad = probe
                    break
                good = probe
            if bad is None:
                raise ArithmeticError(f'Unable to bracket merged-tail onset at K={K}')
            while good-bad > tolerance:
                middle = (good+bad)/2
                if status(middle) > 0:
                    good = middle
                else:
                    bad = middle
            onset = physical_shift+physical_scale*(good+bad)/2
            last_level = int(sp.floor((sp.Rational(str(onset))-shift)/scale-sp.Rational(1, 2)))
            # Exclude the exact levels lying inside the merged region.
            domain_max = physical_shift+physical_scale*Decimal(last_level+1)
            domain_min = physical_shift
            per_depth_cache = directory/f'wide_K{K}.csv' if directory else None
            if last_level < 0:
                bands = pd.DataFrame(columns=['K','level','E_level','E_lower','E_upper',
                                               'left_clipped','right_clipped'])
            else:
                bands = spectral_sequence(potential, [K], Emin=str(domain_min), Emax=str(domain_max),
                                          boundary_digits=boundary_digits, precision=160,
                                          verify_precision=True, progress=progress, cache_path=per_depth_cache,
                                          mass=mass, hbar=hbar)
            frames.append(bands)
            isolated = bands.loc[(bands.left_clipped == False) & (bands.right_clipped == False)]
            summaries.append({'K': K, 'separated_levels': len(isolated),
                              'last_separated_energy': float(isolated.E_level.max()) if len(isolated) else None,
                              'tail_onset': float(onset),
                              'tail_onset_uncertainty': float(physical_scale*(good-bad)/2),
                              'sampled_tail_start': float(physical_shift+physical_scale*Decimal(horizon)),
                              'sampled_tail_end': float(physical_shift+physical_scale*Decimal(horizon+2)),
                              'runtime_seconds': time.perf_counter()-start})
            if progress:
                print(f'K = {K}: {summaries[-1]["separated_levels"]} separated levels; '
                      f'merged tail starts near E = {float(onset):.3f}', flush=True)
        plot_max = physical_shift+physical_scale*Decimal(max(
            int(round((row['sampled_tail_start']-float(shift))/float(scale))) for row in summaries)+tail_padding)
        # Check additional midpoint samples up to the common plotting edge.
        for row in summaries:
            first = int(round((row['sampled_tail_end']-float(shift))/float(scale)))+1
            last = int(round((float(plot_max)-float(shift))/float(scale)))
            for integer in range(first,last+1):
                if _qmb_numeric.sho_numeric_positive(row['K'],str(integer),240)!=1:
                    raise ArithmeticError('Merged tail contains a rejected midpoint; widen the tail search')
            row['plot_Emin'] = float(physical_shift)
            row['plot_Emax'] = float(plot_max)
    result, summary = pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)
    if directory:
        saved.write_text(json.dumps({'parameters': key, 'bands': result.to_dict('records'),
                                    'summary': summary.to_dict('records')}, indent=2))
    return result, summary


def energy_table(bands, potential, K=None, mass=1, hbar=1):
    """Separated quadratic-potential bands in units of omega (hbar=1 by default).

    epsilon_K is the furthest endpoint's absolute distance from theory.
    Endpoints retain the numerical boundary resolution of the input bands.
    The highest available depth is selected unless K is supplied.
    """
    import pandas as pd
    if not isinstance(potential, PolynomialPotential) or potential.degree != 2:
        raise ValueError('Theoretical oscillator levels require a quadratic potential')
    mass, hbar = _rational(mass), _rational(hbar)
    if mass <= 0 or hbar <= 0:
        raise ValueError('mass and hbar must be positive')
    columns = ['Energy Level', 'E_min', 'E_max', 'E_Theoretical', 'Width upper bound', 'epsilon_K']
    if bands.empty:
        return pd.DataFrame(columns=columns)
    K = operator.index(K) if K is not None else int(bands.K.max())
    if K not in set(bands.K):
        raise ValueError(f'No energy bands available for K={K}')
    selected = bands.loc[(bands.K == K) & ~bands.left_clipped & ~bands.right_clipped].sort_values('level')
    from decimal import Decimal, localcontext
    rows = []
    with localcontext() as ctx:
        ctx.prec = 110
        D = lambda value: Decimal(str(value))
        omega = D(sp.N(sp.sqrt(2*potential.coefficients[2]/mass), 110))
        for _, band in selected.iterrows():
            theoretical = D(sp.N(potential.coefficients[0], 110))/omega + D(sp.N(hbar, 110))*(D(int(band.level))+D('0.5'))
            # Keep small offsets separate from the level energy: float endpoints
            # can coincide even when a nonzero interval remains.
            if 'left_offset' in selected.columns:
                left, right = D(band.left_offset)/omega, D(band.right_offset)/omega
            else:
                left = theoretical-D(band.E_lower)/omega
                right = D(band.E_upper)/omega-theoretical
            width = (D(band.left_width_upper)+D(band.right_width_upper))/omega if 'left_width_upper' in selected.columns else left+right
            rows.append({'Energy Level': int(band.level),
                         'E_min': theoretical-left, 'E_max': theoretical+right,
                         'E_Theoretical': theoretical, 'epsilon_K': max(abs(left), abs(right)),
                         'Width upper bound': width})
    return pd.DataFrame(rows, columns=columns)



def plot_spectrum_to_continuum(bands, summary, save_path=None):
    """Publication figure: resolved bands, unresolved tail, and level counts."""
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    if summary.empty:
        raise ValueError("No spectrum results to plot")
    summary = summary.sort_values("K").reset_index(drop=True)
    # Dark blue through teal remains legible against the neutral unresolved tail.
    anchors = np.array([[49, 65, 111], [37, 119, 139], [28, 142, 128]]) / 255
    positions = np.linspace(0, 2, len(summary))
    colors = np.array([np.interp(positions, [0, 1, 2], anchors[:, channel])
                       for channel in range(3)]).T
    ink, muted, grid, tail = '#182b3a', '#647481', '#e8edef', '#e9edef'
    style = {'font.family': 'DejaVu Sans', 'font.size': 10,
             'axes.labelcolor': ink, 'text.color': ink, 'xtick.color': muted,
             'ytick.color': ink, 'pdf.fonttype': 42, 'ps.fonttype': 42,
             'svg.fonttype': 'none', 'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(14, 5.3), facecolor='white')
        layout = fig.add_gridspec(1, 2, width_ratios=[5.6, 1.25],
                                 left=0.075, right=0.975, bottom=0.13,
                                 top=0.96, wspace=0.09)
        ax = fig.add_subplot(layout[0, 0])
        counts = fig.add_subplot(layout[0, 1], sharey=ax)
        xmin, xmax = summary.plot_Emin.min(), summary.plot_Emax.max()
        height = 0.23
        for y, ((_, row), color) in enumerate(zip(summary.iterrows(), colors)):
            # A muted ribbon is an unresolved allowed region, not a density of states.
            ax.add_patch(Rectangle((row.tail_onset, y-height/2),
                                   xmax-row.tail_onset, height,
                                   facecolor=tail, edgecolor='none', zorder=1))
            ax.plot([row.tail_onset, xmax], [y, y], color='#cbd4d9', linewidth=0.8, zorder=2)
            group = bands[bands.K == row.K]
            for _, band in group.iterrows():
                if band.E_upper > band.E_lower:
                    ax.add_patch(Rectangle((band.E_lower, y-height/2),
                                           band.E_upper-band.E_lower, height,
                                           facecolor=color, edgecolor='none', zorder=3))
                ax.plot(band.E_level, y, '|', color=color,
                        markersize=10, markeredgewidth=1.45, zorder=4)
            ax.plot(row.tail_onset, y, 'o', color=color, markersize=5,
                    markeredgecolor='white', markeredgewidth=0.7, zorder=6)
            near_edge = row.tail_onset > xmax-0.16*(xmax-xmin)
            ax.annotate(rf'$E \simeq {row.tail_onset:.2f}$', (row.tail_onset, y),
                        xytext=(-7 if near_edge else 7, 12), textcoords='offset points',
                        ha='right' if near_edge else 'left', va='bottom',
                        color=color, fontsize=9.5)
            counts.barh(y, row.separated_levels, height=height, color=color,
                        alpha=0.9, edgecolor='none')
            counts.text(row.separated_levels+max(1, summary.separated_levels.max()*0.04),
                        y, str(int(row.separated_levels)), va='center',
                        ha='left', color=color, weight='bold', fontsize=12)

        # Join the sampled edges to guide the eye; no extra transition data is inferred.
        ax.plot(summary.tail_onset, np.arange(len(summary)), color='#96a5ae',
                linewidth=1, linestyle=(0, (3, 4)), zorder=0)
        ax.set(xlim=(xmin, xmax), ylim=(-0.45, len(summary)-0.35),
               yticks=range(len(summary)),
               yticklabels=[f'$K = {int(k)}$' for k in summary.K],
               xlabel='Energy $E$', ylabel='Hankel depth')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=3))
        ax.set_axisbelow(True)
        ax.grid(axis='x', color=grid, linewidth=0.7)
        ax.tick_params(axis='both', length=0, pad=9)
        counts.set_xlim(0, max(4, summary.separated_levels.max()*1.26))
        counts.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        counts.set_axisbelow(True)
        counts.grid(axis='x', color=grid, linewidth=0.7)
        counts.tick_params(axis='y', left=False, labelleft=False)
        counts.tick_params(axis='x', length=0, pad=9)
        counts.set_xlabel('Separated levels')
        for panel in (ax, counts):
            for side in ('top', 'right', 'left'):
                panel.spines[side].set_visible(False)
            panel.spines['bottom'].set_color('#cad3d8')
            panel.spines['bottom'].set_linewidth(0.8)
        if save_path:
            path = Path(save_path).with_suffix('.png')
            fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0.15)
        plt.show()
    return fig
