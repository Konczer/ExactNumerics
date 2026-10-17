# QMBootstrap

QMBootstrap computes moment constraints for one-dimensional quantum mechanics.
It combines a Python interface with C++ implementations of moment recurrences,
exact rational calculations, and high-precision matrix positivity tests.
Jupyter provides the working demonstration; the calculation lives in the library.

**Author:** Rithwik Ranganathan

**Contact:** [rithwik.ranganathan@gmail.com](mailto:rithwik.ranganathan@gmail.com)

## What it computes

The Hamiltonian convention is

$$
H=-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V(x).
$$

A potential is specified by its polynomial coefficients. At Hankel depth
$K$, the moment matrix is $M_{ij}=\langle x^{i+j}\rangle$, with
$i,j=0,\ldots,K-1$. Positivity constrains the candidate energies and,
for higher-degree potentials, independent moments.

The library supports even, confining polynomial potentials with rational
coefficients. Its spectrum comparison and automatic energy-range search
currently support quadratic potentials. Quartic and higher-degree calculations
require a search over additional moments.

For a harmonic oscillator, the main notebook compares the energy intervals at
several depths and counts the separated levels. Its high-energy merged region
is a finite-depth resolution limit: the physical oscillator spectrum remains
discrete.

## System requirements

| Component | Requirement |
|---|---|
| Python | 3.10 or later; the existing environment uses Python 3.12 |
| C++ compiler | C++17 support; required when building or rebuilding the extensions |
| Boost | Headers containing `boost/multiprecision`; no compiled Boost libraries required |
| Build dependencies | setuptools 68 or later and pybind11 2.11 or later; declared in `pyproject.toml` |
| Python dependencies | SymPy 1.12 or later, pandas, and matplotlib |
| Notebook dependencies | JupyterLab, ipykernel, and nbclient; installed by the `notebook` extra |

The project has been built and tested on Windows with Visual Studio 2022 MSVC
and Python 3.12. GCC and Clang builds require the same C++17 and Boost header
support, but have not been validated in this workspace. Compiled extensions must
match the Python interpreter and platform used to run them.

Numerical comparisons use 160- and 240-decimal-digit arithmetic. One dense
positivity check takes approximately $O(K^3)$ arithmetic operations and
$O(K^2)$ memory. A spectrum calculation performs many such checks. There is
no fixed upper depth cap in the interface; runtime, memory, and numerical
conditioning limit practical depths.

## Installation

Run these commands from the repository root. `python` must refer to the
interpreter you intend to use.

On Windows, install the Visual Studio C++ build tools and Boost headers first:

```powershell
python -m venv .venv
$env:BOOST_ROOT = 'C:\Libraries\boost'  # This directory must contain boost\.
& .\.venv\Scripts\python.exe -m pip install -e '.[notebook]'
& .\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name qmboot-venv --display-name 'Python (qmboot .venv)'
```

For a Unix-like shell with a suitable compiler:

```sh
python3 -m venv .venv
export BOOST_ROOT=/path/to/boost  # Omit if Boost headers are on the include path.
.venv/bin/python -m pip install -e '.[notebook]'
.venv/bin/python -m ipykernel install --sys-prefix --name qmboot-venv --display-name 'Python (qmboot .venv)'
```

The installation builds `_qmb_native` and `_qmb_numeric_v2`. Their compilation
cost is paid at installation, rather than at each Python call. Reinstall after
changing C++ sources and restart any kernel that has loaded the extensions.

## Use the notebook

Open [QMBootstrap_Fast.ipynb](QMBootstrap_Fast.ipynb) and select
**Python (qmboot .venv)** as the kernel. In the existing workspace, the virtual
environment and VS Code interpreter settings are already configured.

Edit the configuration cell:

```python
Ks = [10, 20, 50]  # Unique integer depths >= 2.
potential = qmboot.PolynomialPotential({2: "1/2"})

plot_boundary_digits = 2
table_boundary_digits = 12
table_relative_tolerance = 1e-5
```

Run the cells in order. The plot compares every depth in `Ks`; the energy table
always uses `max(Ks)`. Start with modest depths before requesting a large run.

The plot uses a boundary resolution of $10^{-2}\omega$. The table starts at
$10^{-12}\omega$ and refines resolved intervals until their combined
boundary-bracket uncertainty is at most $10^{-5}$ of the interval's lower
width estimate. It reuses those brackets and prints progress after each level.
Intervals narrower than the initial resolution remain width upper limits;
refinement also stops at the configured precision limit.

The notebook uses $m=\hbar=1$, with
$\omega=\sqrt{2v_2/m}$ for $V(x)=v_0+v_2x^2$. Table energies and widths
are divided by $\omega$. The ground state has index $n=0$, and
$E_n=v_0+\hbar\omega(n+\tfrac12)$.

The table reports the endpoints, theoretical energy, width upper bound, and

$$
\epsilon_K=\frac{\max\bigl(|E_{\min}-E_{\mathrm{theory}}|,
|E_{\max}-E_{\mathrm{theory}}|\bigr)}{\omega}.
$$

This endpoint error measures agreement with theory without cancellation between
left and right errors. Values display four decimal places in scientific notation;
CSV files retain the underlying decimal values.

### Outputs and caching

| File | Contents |
|---|---|
| `spectrum_extended.png` | Spectrum comparison figure, exported at 300 dpi |
| `energy_levels_K<K>.csv` | Energy table for the largest selected depth |
| `.qmboot-cache/continuum.json` | Combined spectrum data and resolution summary |
| `.qmboot-cache/wide_K<K>.csv` | Numerical bands for each plotted depth |
| `.qmboot-cache/table_adaptive_K<K>.csv` | Detailed adaptive table calculation |

Here `<K>` is the actual depth, for example `energy_levels_K250.csv`. Cache
metadata records the calculation parameters; results are reused only when those
parameters match. Delete the relevant cache file and its sidecar JSON to force
recalculation. An interrupted table calculation does not save a partial table.

If the notebook is edited externally, reload it from disk before executing it.
Restarting a kernel clears Python state but does not refresh an unsaved editor
buffer. A running calculation continues with the settings it started with.

## Use the Python library

```python
import qmboot

potential = qmboot.PolynomialPotential({2: "1/2"})
bands, summary = qmboot.spectrum_to_continuum(
    potential, Ks=[10, 20], boundary_digits=2,
)
figure = qmboot.plot_spectrum_to_continuum(
    bands, summary, save_path="spectrum_extended.png",
)
```

For exact quadratic calculations at small depths:

```python
bootstrap = qmboot.Bootstrap(potential, hankel_depth=10)
intervals = bootstrap.run(Emin=0, Emax=6, root_digits=30, method="exact")
```

`method="exact"` is the default. `method="auto"` selects exact calculations
through K=12 and numerical level-centred searches at larger depths. Exact
polynomial construction and root isolation become expensive at large depths.

For a quartic potential, provide the independent moment $\mu_2$:

```python
quartic = qmboot.Bootstrap(
    qmboot.PolynomialPotential({2: "1/2", 4: 1}), hankel_depth=4,
)
matrix = quartic.hankel(energy=1, seed_moments={2: "1/3"})
samples = quartic.scan(
    energies=["9/10", "1", "11/10"],
    seed_grids={2: ["1/4", "1/3", "2/5"]},
)
```

For an even potential of degree $d$, the independent seed powers are
$2,4,\ldots,d-2$. `scan()` classifies the supplied samples; it does not
establish continuous allowed intervals between them.

## Numerical interpretation

The numerical oscillator solver searches around known theoretical energy levels.
It is suitable for studying finite-depth interval resolution, but does not discover
an unknown spectrum independently. Tail detection uses sampled positivity checks,
not a proof of positivity at every higher energy.

Boundary classifications are checked at higher arithmetic precision. These checks
support numerical consistency; they do not constitute interval-arithmetic
certificates. Requested tolerance is a search stopping criterion, not a guarantee
of total numerical error. Report convergence checks and unresolved upper limits
when using the results in a paper.

The exact quadratic solver uses rational determinant polynomials and root
isolation when the oscillator energy scale is rational. Its positive open cells
do not separately enumerate isolated semidefinite points. The numerical solver
and the sampled polynomial searches should not be described as certified bounds.

## Validation and repository layout

Run the test suite from the project environment:

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests
```

The tests cover moment recurrences, exact and numerical positivity, energy
scaling, boundary agreement, cache behaviour, adaptive refinement, and table
precision. `benchmark.py` compares small exact calculations with the original
symbolic implementation; it is not a benchmark of the large-depth numerical
sweep.

| Path | Purpose |
|---|---|
| `qmboot.py` | Public Python interface, numerical searches, tables, and plotting |
| `native/bootstrap.cpp` | Exact rational moment and determinant calculations |
| `native/numeric.cpp` | High-precision numerical positivity checks |
| `QMBootstrap_Fast.ipynb` | Main spectrum and energy-table demonstration |
| `QMBootstrap_Polynomial.ipynb` | Additional polynomial demonstration |
| `QMBootstrap_Reference.ipynb` | Original symbolic implementation used by the benchmark |
| `qmb_fast.py` | Compatibility interface for the original oscillator API |
| `run_bootstrap.py` | Command-line runner for small exact calculations |
| `tests/` | Regression and numerical validation tests |
