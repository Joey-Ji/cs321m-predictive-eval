"""Explore measurement-db to find new benchmarks for training data augmentation.

This script downloads and analyzes the full MDB dataset to identify:
1. Which benchmarks are NOT in our current training data
2. Size and quality metrics for each benchmark
3. Recommendations for which benchmarks to add

Usage:
    python scripts/explore_mdb.py
"""

import pandas as pd
from pathlib import Path
import sys

# Current benchmarks in our training data (from joined.parquet)
CURRENT_BENCHMARKS = {
    'mmbench_v11',
    'ai2d_test',
    'mmlupro',
    'rewardbench',
    'bfcl',
    'livecodebench',
    'mathvista_mini',
    'afrimedqa',
    'ultrafeedback',
    'matharena',
    'agentdojo',
    'swebench',
    'hle',
    'mtbench',
    'cybench',
    'androidworld',
}

def load_mdb():
    """Load measurement-db metadata from HuggingFace.

    Instead of loading the full 1.5M row dataset, we load the benchmarks.parquet
    metadata file which has summary statistics for each benchmark.
    """
    print("Loading measurement-db metadata from HuggingFace...")

    try:
        from huggingface_hub import hf_hub_download

        # Download the benchmarks metadata file
        benchmarks_file = hf_hub_download(
            repo_id='aims-foundations/measurement-db',
            filename='benchmarks.parquet',
            repo_type='dataset'
        )

        df = pd.read_parquet(benchmarks_file)
        print(f"✓ Loaded {len(df):,} benchmarks from measurement-db")
        return df
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nMake sure you have installed: pip install huggingface-hub pandas pyarrow")
        sys.exit(1)

def analyze_benchmarks(df):
    """Analyze benchmark statistics."""
    print("\n" + "="*80)
    print("BENCHMARK ANALYSIS")
    print("="*80)

    # Overall stats
    print(f"\n📊 Overall Statistics:")
    print(f"  Total rows: {len(df):,}")
    print(f"  Unique benchmarks: {df['benchmark_id'].nunique()}")
    print(f"  Unique subjects (models): {df['subject_id'].nunique()}")
    print(f"  Unique items: {df['item_id'].nunique():,}")

    # Per-benchmark stats
    benchmark_stats = df.groupby('benchmark_id').agg({
        'item_id': 'nunique',
        'subject_id': 'nunique',
        'response': ['count', 'mean', 'std']
    })

    # Flatten column names
    benchmark_stats.columns = ['n_items', 'n_subjects', 'n_rows', 'avg_response', 'std_response']
    benchmark_stats = benchmark_stats.sort_values('n_rows', ascending=False)

    # Add quality score
    # Quality = (# items) * (# subjects) / 1000, normalized
    benchmark_stats['quality_score'] = (
        (benchmark_stats['n_items'] * benchmark_stats['n_subjects']) / 1000
    )

    return benchmark_stats

def identify_new_benchmarks(benchmark_stats):
    """Identify benchmarks not in current training data."""
    print("\n" + "="*80)
    print("NEW BENCHMARKS AVAILABLE")
    print("="*80)

    # Split into current vs new
    current_mask = benchmark_stats.index.isin(CURRENT_BENCHMARKS)
    current_stats = benchmark_stats[current_mask]
    new_stats = benchmark_stats[~current_mask]

    print(f"\n✓ Currently using: {len(current_stats)} benchmarks")
    print(f"✨ Available to add: {len(new_stats)} benchmarks")
    print(f"   → Potential {len(new_stats) / len(current_stats):.1f}x data expansion!")

    return current_stats, new_stats

def print_top_benchmarks(stats, title, n=20):
    """Print top N benchmarks by quality score."""
    print(f"\n{title}")
    print("-" * 100)
    print(f"{'Benchmark':<40} {'Items':>8} {'Subjects':>8} {'Rows':>10} {'Avg Resp':>10} {'Quality':>10}")
    print("-" * 100)

    for idx, row in stats.head(n).iterrows():
        print(f"{idx:<40} {row['n_items']:>8,} {row['n_subjects']:>8} {row['n_rows']:>10,} "
              f"{row['avg_response']:>10.3f} {row['quality_score']:>10.1f}")

def recommend_benchmarks(new_stats, top_n=20):
    """Recommend which benchmarks to add first."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Criteria for good benchmarks:
    # 1. Quality score (items × subjects)
    # 2. Reasonable size (not too small, not too huge)
    # 3. Good subject diversity (>20 models)

    # Filter: at least 100 items, at least 10 subjects
    recommended = new_stats[
        (new_stats['n_items'] >= 100) &
        (new_stats['n_subjects'] >= 10)
    ].copy()

    # Sort by quality score
    recommended = recommended.sort_values('quality_score', ascending=False)

    print(f"\n🎯 Top {top_n} Recommended Benchmarks to Add:")
    print(f"   (Filtered: ≥100 items, ≥10 subjects)")
    print()

    print_top_benchmarks(recommended, "", n=top_n)

    print(f"\n📈 Impact if you add top {top_n}:")
    top_20 = recommended.head(top_n)
    print(f"   Additional items: {top_20['n_items'].sum():,}")
    print(f"   Additional rows: {top_20['n_rows'].sum():,}")
    print(f"   Average quality score: {top_20['quality_score'].mean():.1f}")

    return recommended

def save_recommendations(recommended, output_path):
    """Save recommendations to CSV."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    recommended.to_csv(output_file)
    print(f"\n💾 Saved detailed analysis to: {output_file}")
    print(f"   You can review this file to select which benchmarks to add.")

def main():
    """Main exploration workflow."""
    print("\n" + "="*80)
    print("MEASUREMENT-DB EXPLORATION")
    print("="*80)
    print("\nThis script will:")
    print("  1. Download the full measurement-db dataset")
    print("  2. Analyze all available benchmarks")
    print("  3. Identify new benchmarks not in your current data")
    print("  4. Recommend which benchmarks to add for best improvement")

    # Load data
    df = load_mdb()

    # Analyze
    benchmark_stats = analyze_benchmarks(df)

    # Identify new benchmarks
    current_stats, new_stats = identify_new_benchmarks(benchmark_stats)

    # Print current benchmarks
    print_top_benchmarks(current_stats, "\n📦 Current Benchmarks in Training Data:", n=len(current_stats))

    # Print all new benchmarks
    print_top_benchmarks(new_stats, f"\n✨ All {len(new_stats)} New Benchmarks Available:", n=30)

    # Recommend best benchmarks
    recommended = recommend_benchmarks(new_stats, top_n=20)

    # Save to CSV
    output_path = "data/mdb_new_benchmarks.csv"
    save_recommendations(recommended, output_path)

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Review the recommendations above")
    print("2. Check data/mdb_new_benchmarks.csv for full details")
    print("3. Select 5-10 benchmarks to add (start small)")
    print("4. Use scripts/augment_training_data.py to merge them into your data")
    print("5. Retrain Stage 1 + Stage 2 and evaluate!")
    print("\n✨ Good luck!\n")

if __name__ == "__main__":
    main()
