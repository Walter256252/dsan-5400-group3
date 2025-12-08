"""
stats.py

Statistical tests used in analysis_runner:
- t-test
- Mann–Whitney U test
- Kolmogorov–Smirnov test
- Chi-square test for categorical RoBERTa labels
"""

import numpy as np
import pandas as pd
from scipy import stats


def run_all_numeric_tests(df: pd.DataFrame, gender_a: str, gender_b: str) -> pd.DataFrame:
    numeric_cols = [
        col for col in [
            "vader_compound",
            "textblob_polarity",
            "textblob_subjectivity",
            "male_pronoun_count",
            "female_pronoun_count",
            "article_length_words",
        ] if col in df.columns
    ]

    rows = []
    for col in numeric_cols:
        sub = df[[col, "gender"]].dropna()
        a_vals = sub.loc[sub.gender == gender_a, col]
        b_vals = sub.loc[sub.gender == gender_b, col]

        res = {
            "variable": col,
            "n_a": len(a_vals),
            "n_b": len(b_vals),
            "mean_a": a_vals.mean(),
            "mean_b": b_vals.mean(),
        }

        if len(a_vals) > 1 and len(b_vals) > 1:
            res["t_pvalue"] = stats.ttest_ind(a_vals, b_vals, equal_var=False).pvalue
            res["mw_pvalue"] = stats.mannwhitneyu(a_vals, b_vals, alternative="two-sided").pvalue
            res["ks_pvalue"] = stats.ks_2samp(a_vals, b_vals).pvalue
        else:
            res["t_pvalue"] = np.nan
            res["mw_pvalue"] = np.nan
            res["ks_pvalue"] = np.nan

        rows.append(res)

    return pd.DataFrame(rows)


def run_chi_square_test(df: pd.DataFrame, gender_a: str, gender_b: str) -> pd.DataFrame:
    if "roberta_label_str" not in df.columns:
        return pd.DataFrame([{"chi2": np.nan, "pvalue": np.nan, "dof": np.nan}])

    sub = df[df.gender.isin([gender_a, gender_b])]
    contingency = pd.crosstab(sub.gender, sub.roberta_label_str)

    chi2, p, dof, _ = stats.chi2_contingency(contingency)

    return pd.DataFrame([{"chi2": chi2, "pvalue": p, "dof": dof}])
