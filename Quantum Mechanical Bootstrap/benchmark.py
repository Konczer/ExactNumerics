"""Compare cold full runs against the original notebook without plotting."""
import argparse
import json
import time
from pathlib import Path
import qmb_fast

parser = argparse.ArgumentParser()
parser.add_argument("--ks", nargs="+", type=int, default=[6, 8, 10])
parser.add_argument("--digits", type=int, default=30)
args = parser.parse_args()
notebook = json.loads(Path("QMBootstrap_Reference.ipynb").read_text(encoding="utf-8-sig"))
reference = {}
for index in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
    exec("".join(notebook["cells"][index]["source"]), reference)
for K in args.ks:
    start = time.perf_counter()
    baseline = reference["bootstrap_dataframe"](K, root_digits=args.digits)
    old_time = time.perf_counter() - start
    start = time.perf_counter()
    accelerated = qmb_fast.bootstrap_dataframe(K, root_digits=args.digits)
    new_time = time.perf_counter() - start
    import pandas as pd
    pd.testing.assert_frame_equal(baseline, accelerated)
    print(f"K={K}: original={old_time:.3f}s C++={new_time:.3f}s speedup={old_time/new_time:.2f}x")
