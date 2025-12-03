#!/usr/bin/env python
"""
run_stats.py

Statistical tests for gender differences in sentiment and pronoun usage
in Wikipedia biographies.

Tests performed (for two genders, default: male vs female):

- For continuous variables (e.g., vader_compound, textblob_polarity):
    * mean / std / n by group
    * independent two-sample t-test
    * Mann–Whitney U test (nonparametric)
    * Kolmogorov–Smirnov test (distributional difference)

- For RoBERTa sentiment labels:
    * Chi-square test of independence on label distribution by gender

Results are printed to stdout and saved to CSVs in results/sentiment/.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "biographies_with_sentiment.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "sentiment"


# CLI 

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run statistical tests on sentiment-enriched biographies."
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=str(DEFAULT_INPUT),
        help=f"Path to sentiment-enriched CSV (default: {DEFAULT_INPUT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save stats tables (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Optional: use only the first N rows (for quick testing).",
    )
    parser.add_argument(
        "--gender-a",
        type=str,
        default="male",
        help="First gender group for comparison (default: 'male').",
    )
    parser.add_argument(
        "--gender-b",
        type=str,
        default="female",
        help="Second gender group for comparison (default: 'female').",
    )

    return parser.parse_args()


# Helpers

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path, sample_n: Optional[int]) -> pd.DataFrame:
    print(">>> run_stats.py is running <<<")
    print(f"[INFO] Loading from: {path}")

    if sample_n is not None:
        print(f"[INFO] Using sample-n = {sample_n}")
        df = pd.read_csv(path, nrows=sample_n)
    else:
        print("[INFO] Using FULL dataset")
        df = pd.read_csv(path)

    print(f"[INFO] Loaded shape: {df.shape}")

    # Map RoBERTa labels to strings if present
    label_map = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    if "roberta_label" in df.columns:
        df["roberta_label_str"] = df["roberta_label"].map(label_map).fillna(
            df["roberta_label"]
        )

    print("\n[INFO] First 3 rows:")
    print(df.head(3))

    return df


def two_group_tests(
    df: pd.DataFrame,
    col: str,
    group_col: str,
    group_a: str,
    group_b: str,
) -> Dict[str, float]:
    """
    Run t-test, Mann-Whitney U, and KS test for a numeric column
    between two groups.
    """
    sub = df[[group_col, col]].dropna()
    a_vals = sub.loc[sub[group_col] == group_a, col].astype(float)
    b_vals = sub.loc[sub[group_col] == group_b, col].astype(float)

    results: Dict[str, float] = {
        "group_a": group_a,
        "group_b": group_b,
        "n_a": len(a_vals),
        "n_b": len(b_vals),
        "mean_a": float(a_vals.mean()) if len(a_vals) else np.nan,
        "mean_b": float(b_vals.mean()) if len(b_vals) else np.nan,
        "std_a": float(a_vals.std(ddof=1)) if len(a_vals) > 1 else np.nan,
        "std_b": float(b_vals.std(ddof=1)) if len(b_vals) > 1 else np.nan,
    }

    if len(a_vals) > 1 and len(b_vals) > 1:
        # t-test 
        t_stat, t_p = stats.ttest_ind(a_vals, b_vals, equal_var=False)
        results["t_stat"] = float(t_stat)
        results["t_pvalue"] = float(t_p)

        # Mann–Whitney U 
        try:
            u_stat, u_p = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            results["mw_stat"] = float(u_stat)
            results["mw_pvalue"] = float(u_p)
        except ValueError:
            results["mw_stat"] = np.nan
            results["mw_pvalue"] = np.nan

        # KS test on distributions
        ks_stat, ks_p = stats.ks_2samp(a_vals, b_vals)
        results["ks_stat"] = float(ks_stat)
        results["ks_pvalue"] = float(ks_p)
    else:
        # Not enough data
        results["t_stat"] = np.nan
        results["t_pvalue"] = np.nan
        results["mw_stat"] = np.nan
        results["mw_pvalue"] = np.nan
        results["ks_stat"] = np.nan
        results["ks_pvalue"] = np.nan

    return results


def chi_square_roberta(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
    label_col: str = "roberta_label_str",
) -> Dict[str, float]:
    """
    Chi-square test of independence between gender and RoBERTa label distribution.
    Uses only the two specified gender groups.
    """
    sub = df[[group_col, label_col]].dropna()
    sub = sub[sub[group_col].isin([group_a, group_b])]

    contingency = pd.crosstab(sub[group_col], sub[label_col])
    print("\n[INFO] RoBERTa contingency table used for chi-square:")
    print(contingency)

    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    return {
        "chi2": float(chi2),
        "pvalue": float(p),
        "dof": int(dof),
    }


# Main 

def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    df = load_data(input_path, sample_n=args.sample_n)

    if "gender" not in df.columns:
        raise ValueError("No 'gender' column found; cannot run gender-based tests.")

    gender_a = args.gender_a
    gender_b = args.gender_b

    print(f"\n[INFO] Comparing groups: {gender_a!r} vs {gender_b!r}")

    # Continuous sentiment variables
    numeric_cols: List[str] = [
        "vader_compound",
        "textblob_polarity",
        "textblob_subjectivity",
    ]

    # Add pronoun-related columns if present
    for extra in [
        "male_pronoun_count",
        "female_pronoun_count",
        "article_length_words",
    ]:
        if extra in df.columns:
            numeric_cols.append(extra)

    stats_rows = []
    for col in numeric_cols:
        if col not in df.columns:
            continue
        print(f"\n[INFO] Running tests for variable: {col}")
        res = two_group_tests(df, col, "gender", gender_a, gender_b)
        res["variable"] = col
        stats_rows.append(res)

        # Print to terminal
        print(f"  n_{gender_a} = {res['n_a']}, mean_{gender_a} = {res['mean_a']:.4f}")
        print(f"  n_{gender_b} = {res['n_b']}, mean_{gender_b} = {res['mean_b']:.4f}")
        print(f"  t-stat = {res['t_stat']:.4f}, p = {res['t_pvalue']:.3e}")
        print(f"  MW U  = {res['mw_stat']:.4f}, p = {res['mw_pvalue']:.3e}")
        print(f"  KS    = {res['ks_stat']:.4f}, p = {res['ks_pvalue']:.3e}")

    stats_df = pd.DataFrame(stats_rows).set_index("variable")
    out_numeric = output_dir / f"stats_continuous_{gender_a}_vs_{gender_b}.csv"
    stats_df.to_csv(out_numeric)
    print(f"\n[INFO] Saved continuous-variable stats to: {out_numeric}")

    # Chi-square for RoBERTa label distribution
    if "roberta_label_str" in df.columns:
        print("\n[INFO] Running chi-square test for RoBERTa label distribution...")
        chi_res = chi_square_roberta(df, "gender", gender_a, gender_b)

        chi_df = pd.DataFrame([chi_res])
        out_chi = output_dir / f"stats_chi2_roberta_{gender_a}_vs_{gender_b}.csv"
        chi_df.to_csv(out_chi, index=False)
        print(f"[INFO] Saved chi-square results to: {out_chi}")
        print(
            f"  chi2 = {chi_res['chi2']:.4f}, dof = {chi_res['dof']}, "
            f"p = {chi_res['pvalue']:.3e}"
        )
    else:
        print("\n[WARN] 'roberta_label_str' not found; skipping chi-square test.")

    print("\n>>> run_stats.py finished successfully <<<")


if __name__ == "__main__":
    main()
