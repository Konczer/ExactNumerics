"""Run the bootstrap from a terminal; Jupyter is optional."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ks", nargs="+", type=int, default=[6, 8, 10, 12, 14, 16])
    parser.add_argument("--emin", type=str, default="0")
    parser.add_argument("--emax", type=str, default="6")
    parser.add_argument("--digits", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("bootstrap_results.csv"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    import qmboot
    import pandas as pd
    potential = qmboot.PolynomialPotential({2: "1/2"})
    results = pd.concat([
        qmboot.Bootstrap(potential, hankel_depth=K).run(args.emin, args.emax, args.digits)
        for K in args.ks
    ], ignore_index=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))
    print(f"Saved {args.output.resolve()}")
    if args.plot:
        qmboot.plot_regions(results)


if __name__ == "__main__":
    main()
