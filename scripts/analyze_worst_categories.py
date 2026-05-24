#!/usr/bin/env python3
"""
Analyze worst performing categories from Modal evaluation results.
Find which domains need more training data.
"""

# Per-category results from Modal evaluation (canon_v2 baseline)
MODAL_RESULTS = {
    "afrimedqa::source=afrimedqa-v1|prompt=instruct": -0.7506920908080708,
    "afrimedqa::source=afrimedqa-v2|prompt=base": -0.6214945984901696,
    "afrimedqa::source=unknown|prompt=base": -0.6525285612153264,
    "rewardbench::subset=llmbar-adver-neighbor": -0.7737358096907058,
    "mmbench_v11::skill=attribute_reasoning": -0.729552408904823,
    "mmlupro::none": -0.7240625093420522,
    "rewardbench::subset=llmbar-adver-GPTOut": -0.6962970884616275,
    "agentdojo::metric=utility|attack=important_instructions_no_user_name": -0.6798937658496075,
    "agentdojo::metric=utility|attack=important_instructions_no_names": -0.6986978131300796,
    "agentdojo::metric=security|attack=important_instructions_no_user_name": -0.6985177276386969,
    "agentdojo::metric=security|attack=important_instructions_no_names": -0.6873229705267263,
    "agentdojo::metric=security|attack=important_instructions_no_model_name": -0.6788291329945012,
    "agentdojo::metric=security|attack=tool_knowledge": -0.6723735189243039,
    "agentdojo::metric=utility|attack=important_instructions_no_model_name": -0.6697223331814728,
    "rewardbench::subset=llmbar-adver-manual": -0.6278492715785475,
    "ai2d_test::none": -0.5961003009792853,
    "agentdojo::metric=security|attack=important_instructions_wrong_model_name": -0.5951529762607157,
    "bfcl::none": -0.5890155829742626,
    "swebench::none": -0.5666567008857034,
    "agentdojo::metric=utility|attack=captcha_dos": -0.5641549637221105,
    "hle::none": -0.5569536797703227,
    "agentdojo::metric=utility|attack=offensive_email_dos": -0.5550117599968857,
    "agentdojo::metric=utility|attack=none": -0.5543380470885778,
    "livecodebench::source=submissions": -0.5452455665326087,
    "agentdojo::metric=security|attack=captcha_dos": -0.5421865208478839,
    "agentdojo::metric=utility|attack=important_instructions": -0.5416257828912109,
    "donotanswer": -0.5328937260620861,
    "agentdojo::metric=security|attack=offensive_email_dos": -0.5313274926165995,
    "rewardbench::subset=llmbar-adver-GPTInst": -0.5269869123942287,
    "agentdojo::metric=utility|attack=dos": -0.5218294395908827,
    "agentdojo::metric=utility|attack=direct": -0.4925813272406375,
    "mathvista_mini::none": -0.5041750196325923,
    "matharena::none": -0.5391692559729004,
}

def analyze_worst_categories():
    """Find worst performing categories and group by benchmark."""

    # Sort by MLL (worst first)
    sorted_cats = sorted(MODAL_RESULTS.items(), key=lambda x: x[1])

    print("="*80)
    print("TOP 20 WORST PERFORMING CATEGORIES (by MLL)")
    print("="*80)
    print()

    for i, (category, mll) in enumerate(sorted_cats[:20], 1):
        benchmark = category.split("::")[0]
        condition = "::".join(category.split("::")[1:]) if "::" in category else "default"
        print(f"{i:2d}. MLL={mll:7.4f}  [{benchmark:20s}] {condition}")

    print()
    print("="*80)
    print("WORST PERFORMING BENCHMARKS (aggregated)")
    print("="*80)
    print()

    # Aggregate by benchmark
    benchmark_scores = {}
    for category, mll in MODAL_RESULTS.items():
        benchmark = category.split("::")[0]
        if benchmark not in benchmark_scores:
            benchmark_scores[benchmark] = []
        benchmark_scores[benchmark].append(mll)

    benchmark_avg = {b: sum(scores)/len(scores) for b, scores in benchmark_scores.items()}
    sorted_benchmarks = sorted(benchmark_avg.items(), key=lambda x: x[1])

    for i, (benchmark, avg_mll) in enumerate(sorted_benchmarks[:15], 1):
        n_categories = len(benchmark_scores[benchmark])
        worst_cat_mll = min(benchmark_scores[benchmark])
        print(f"{i:2d}. Avg MLL={avg_mll:7.4f}  Worst={worst_cat_mll:7.4f}  {benchmark:20s} ({n_categories} categories)")

    print()
    print("="*80)
    print("DOMAIN ANALYSIS")
    print("="*80)
    print()

    # Categorize by domain
    domains = {
        "Medical/Health": ["afrimedqa"],
        "Vision/Multimodal": ["ai2d_test", "mmbench_v11", "mathvista_mini"],
        "Code/Programming": ["livecodebench", "bfcl", "swebench", "hle"],
        "Math/Reasoning": ["mmlupro", "matharena"],
        "Security/Adversarial": ["agentdojo"],
        "LLM Evaluation": ["rewardbench"],
    }

    domain_scores = {}
    for domain, benchmarks in domains.items():
        scores = []
        for category, mll in MODAL_RESULTS.items():
            benchmark = category.split("::")[0]
            if benchmark in benchmarks:
                scores.append(mll)
        if scores:
            domain_scores[domain] = {
                "avg": sum(scores)/len(scores),
                "worst": min(scores),
                "count": len(scores)
            }

    sorted_domains = sorted(domain_scores.items(), key=lambda x: x[1]["avg"])

    for i, (domain, stats) in enumerate(sorted_domains, 1):
        print(f"{i}. {domain:25s}  Avg MLL={stats['avg']:7.4f}  Worst={stats['worst']:7.4f}  ({stats['count']} categories)")

    print()
    print("="*80)
    print("RECOMMENDATIONS FOR DATA SEARCH")
    print("="*80)
    print()

    recommendations = [
        ("Medical/Health (AfriMedQA)", "Search: medical QA, healthcare benchmarks, clinical evaluation datasets"),
        ("Vision Tasks (MMBench)", "Search: vision-language datasets, VQA, visual reasoning benchmarks"),
        ("MMLU-like (MMLUPro)", "Search: knowledge QA, MMLU variants, subject exams datasets"),
        ("Adversarial/Security", "Search: prompt injection datasets, red-teaming benchmarks, robustness tests"),
        ("Preference/Ranking", "Search: human preference datasets, reward modeling, pairwise comparison"),
    ]

    for i, (area, query) in enumerate(recommendations, 1):
        print(f"{i}. {area}")
        print(f"   → {query}")
        print()

if __name__ == "__main__":
    analyze_worst_categories()
