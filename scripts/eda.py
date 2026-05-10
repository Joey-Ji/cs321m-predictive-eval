"""Quick EDA over the joined training set.

Run after scripts/download_data.py. Prints a one-page summary:
  - row counts, unique counts (subjects, items, benchmarks, conditions)
  - label balance overall and per-benchmark
  - per-benchmark difficulty (mean correct rate)
  - subject_content metadata field coverage
  - item_content / subject_content length distributions

Output is plain text to stdout; redirect to data/eda_summary.txt if desired.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    sxs = sorted(xs)
    k = max(0, min(len(sxs) - 1, int(q * (len(sxs) - 1))))
    return sxs[k]


def main(joined_path: Path) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(joined_path)
    n_rows = table.num_rows
    print(f"Joined rows: {n_rows:,}")

    subjects = table.column("subject_id").to_pylist()
    items = table.column("item_id").to_pylist()
    benchmarks = table.column("benchmark").to_pylist()
    conditions = table.column("condition").to_pylist()
    labels = table.column("label").to_pylist()
    subject_contents = table.column("subject_content").to_pylist()
    item_contents = table.column("item_content").to_pylist()

    print(f"  unique subjects   : {len(set(subjects)):>8,}")
    print(f"  unique items      : {len(set(items)):>8,}")
    print(f"  unique benchmarks : {len(set(benchmarks)):>8,}")
    print(f"  unique conditions : {len(set(conditions)):>8,}")

    print(f"\nOverall label balance:")
    print(f"  P(label=1) = {mean(labels):.4f}  (n={n_rows:,})")

    print(f"\nPer-benchmark difficulty (top 25 by row count):")
    bench_counter = Counter(benchmarks)
    bench_correct = {b: [] for b in bench_counter}
    for b, l in zip(benchmarks, labels):
        bench_correct[b].append(l)
    rows = sorted(bench_counter.items(), key=lambda kv: -kv[1])[:25]
    print(f"  {'benchmark':<30} {'n':>10} {'P(correct)':>12}")
    for b, n in rows:
        print(f"  {b:<30} {n:>10,} {mean(bench_correct[b]):>12.4f}")

    print(f"\nCondition distribution (top 10):")
    cond_counter = Counter(conditions)
    for c, n in cond_counter.most_common(10):
        print(f"  {c:<30} {n:>10,}")

    print(f"\nsubject_content metadata coverage (line prefixes seen):")
    prefix_counter = Counter()
    for sc in subject_contents:
        if not sc:
            continue
        for line in sc.splitlines():
            if ":" in line:
                prefix = line.split(":", 1)[0].strip()
                prefix_counter[prefix] += 1
    for p, n in prefix_counter.most_common():
        print(f"  {p:<20} {n:>10,}  ({100 * n / n_rows:.1f}%)")

    print(f"\nText length distributions (chars):")
    for name, xs in (
        ("subject_content", [len(s or "") for s in subject_contents]),
        ("item_content", [len(s or "") for s in item_contents]),
    ):
        print(
            f"  {name:<18} mean={mean(xs):>7.0f} median={median(xs):>6.0f} "
            f"p90={quantile(xs, 0.9):>6.0f} p99={quantile(xs, 0.99):>7.0f} "
            f"max={max(xs):>7,}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", default="data/joined.parquet", type=Path)
    args = parser.parse_args()
    main(args.joined)
