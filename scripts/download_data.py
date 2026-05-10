"""Download and cache the public training dataset.

Run once locally (or on Modal). Caches Parquet tables under data/ and
materializes a joined long-format dataset matching the four-field shape that
predict() sees at test time.

Follows the explicit-files loader recipe from README.md to avoid mixing
response tables with *_traces.parquet.

Outputs (under data/):
  - responses.parquet    — long-format response rows
  - items.parquet        — item registry
  - subjects.parquet     — subject registry
  - benchmarks.parquet   — benchmark registry
  - joined.parquet       — training rows in {benchmark, condition,
                          subject_content, item_content, label} shape
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ID = "aims-foundations/measurement-db"
REGISTRY_FILES = {"subjects.parquet", "items.parquet", "benchmarks.parquet"}


def render_subject_content(subject: dict, fallback_subject_id: str) -> str:
    display_name = subject.get("display_name") or fallback_subject_id
    lines = [f"Name: {display_name}"]
    for key, label in (
        ("provider", "Organization"),
        ("params", "Parameters"),
        ("release_date", "Released"),
        ("family", "Family"),
    ):
        value = subject.get(key)
        if value:
            lines.append(f"{label}: {value}")
    return "\n".join(lines)


def main(out_dir: Path, binary_only: bool) -> None:
    from datasets import Features, Value, load_dataset
    from huggingface_hub import HfApi

    out_dir.mkdir(parents=True, exist_ok=True)

    repo_files = HfApi().list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    response_files = sorted(
        name
        for name in repo_files
        if name.endswith(".parquet")
        and name not in REGISTRY_FILES
        and not name.endswith("_traces.parquet")
    )
    print(f"Found {len(response_files)} response tables; skipping traces.")

    response_features = Features(
        {
            "subject_id": Value("string"),
            "item_id": Value("string"),
            "benchmark_id": Value("string"),
            "trial": Value("int64"),
            "test_condition": Value("string"),
            "response": Value("float64"),
            "correct_answer": Value("string"),
            "trace": Value("string"),
        }
    )

    print("Loading responses ...")
    responses = load_dataset(
        REPO_ID,
        data_files=response_files,
        features=response_features,
        split="train",
    )
    print("Loading registry tables ...")
    items = load_dataset(REPO_ID, data_files="items.parquet", split="train")
    subjects = load_dataset(REPO_ID, data_files="subjects.parquet", split="train")
    benchmarks = load_dataset(REPO_ID, data_files="benchmarks.parquet", split="train")

    print(f"  responses : {len(responses):>10,}")
    print(f"  items     : {len(items):>10,}")
    print(f"  subjects  : {len(subjects):>10,}")
    print(f"  benchmarks: {len(benchmarks):>10,}")

    print("Caching to local Parquet ...")
    responses.to_parquet(out_dir / "responses.parquet")
    items.to_parquet(out_dir / "items.parquet")
    subjects.to_parquet(out_dir / "subjects.parquet")
    benchmarks.to_parquet(out_dir / "benchmarks.parquet")

    print("Building joined four-field training rows ...")
    items_by_id = {row["item_id"]: row for row in items}
    subjects_by_id = {row["subject_id"]: row for row in subjects}
    benchmarks_by_id = {row["benchmark_id"]: row for row in benchmarks}

    def to_training_example(row):
        item = items_by_id.get(row["item_id"], {})
        subject = subjects_by_id.get(row["subject_id"], {})
        benchmark = benchmarks_by_id.get(row["benchmark_id"], {})
        return {
            "benchmark": benchmark.get("benchmark_id") or row["benchmark_id"],
            "condition": row["test_condition"] or "none",
            "subject_content": render_subject_content(subject, row["subject_id"]),
            "item_content": item.get("content"),
            "label": row["response"],
            "subject_id": row["subject_id"],
            "item_id": row["item_id"],
        }

    joined = responses.map(to_training_example, remove_columns=responses.column_names)

    if binary_only:
        n_before = len(joined)
        joined = joined.filter(lambda r: r["label"] in (0.0, 1.0))
        print(f"  binary-only filter: {n_before:,} -> {len(joined):,} rows")

    joined.to_parquet(out_dir / "joined.parquet")
    print(f"Wrote {out_dir / 'joined.parquet'} with {len(joined):,} rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data", type=Path)
    parser.add_argument(
        "--binary-only",
        action="store_true",
        default=True,
        help="Drop continuous-scored rows; keep only label in {0,1}.",
    )
    args = parser.parse_args()
    main(args.out, args.binary_only)
