"""
Tests for analysis module.

Tests cover:
- Statistical tests (t-test, Mann-Whitney U, KS test, chi-square)
- Summary statistics generation
- Distribution comparisons
"""

import pytest
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats

from dsan_5400_group3.analysis.stats import (
    run_all_numeric_tests,
    run_chi_square_test
)
from dsan_5400_group3.analysis.summaries import (
    compute_sentiment_summary,
    compute_roberta_distribution,
    compute_pronoun_summary
)


class TestStatisticalTests:
    """Test statistical testing functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            "gender": ["male"] * 100 + ["female"] * 100,
            "vader_compound": np.concatenate([
                np.random.normal(0.5, 0.2, 100),  # Male: mean=0.5
                np.random.normal(0.7, 0.2, 100)   # Female: mean=0.7
            ]),
            "textblob_polarity": np.concatenate([
                np.random.normal(0.08, 0.1, 100),
                np.random.normal(0.10, 0.1, 100)
            ]),
            "textblob_subjectivity": np.concatenate([
                np.random.normal(0.32, 0.05, 100),
                np.random.normal(0.34, 0.05, 100)
            ]),
            "male_pronoun_count": np.concatenate([
                np.random.poisson(20, 100),
                np.random.poisson(2, 100)
            ]),
            "female_pronoun_count": np.concatenate([
                np.random.poisson(1, 100),
                np.random.poisson(22, 100)
            ]),
            "article_length_words": np.concatenate([
                np.random.normal(1478, 200, 100).astype(int),
                np.random.normal(1549, 200, 100).astype(int)
            ])
        })
    
    def test_run_all_numeric_tests_basic(self, sample_data):
        """Test that run_all_numeric_tests returns expected structure."""
        result = run_all_numeric_tests(sample_data, gender_a="male", gender_b="female")
        
        assert isinstance(result, pd.DataFrame)
        
        # Should have results for all numeric columns
        expected_cols = [
            "vader_compound", 
            "textblob_polarity",
            "textblob_subjectivity",
            "male_pronoun_count",
            "female_pronoun_count",
            "article_length_words"
        ]
        assert len(result) == len(expected_cols)
        
        # Should have all expected columns
        assert "variable" in result.columns
        assert "n_a" in result.columns
        assert "n_b" in result.columns
        assert "mean_a" in result.columns
        assert "mean_b" in result.columns
        assert "t_pvalue" in result.columns
        assert "mw_pvalue" in result.columns
        assert "ks_pvalue" in result.columns
    
    def test_run_all_numeric_tests_detects_difference(self, sample_data):
        """Test that tests detect significant differences."""
        result = run_all_numeric_tests(sample_data, gender_a="male", gender_b="female")
        
        # vader_compound has clear difference (0.5 vs 0.7)
        vader_row = result[result.variable == "vader_compound"].iloc[0]
        
        # Female mean should be higher
        assert vader_row["mean_b"] > vader_row["mean_a"]
        
        # All tests should detect significant difference (p < 0.05)
        assert vader_row["t_pvalue"] < 0.05
        assert vader_row["mw_pvalue"] < 0.05
        assert vader_row["ks_pvalue"] < 0.05
    
    def test_run_all_numeric_tests_sample_sizes(self, sample_data):
        """Test that sample sizes are reported correctly."""
        result = run_all_numeric_tests(sample_data, gender_a="male", gender_b="female")
        
        # All rows should have n_a = 100 and n_b = 100
        assert all(result["n_a"] == 100)
        assert all(result["n_b"] == 100)
    
    def test_run_all_numeric_tests_handles_missing_columns(self):
        """Test handling of missing columns."""
        df = pd.DataFrame({
            "gender": ["male", "female"],
            "vader_compound": [0.5, 0.7]
        })
        
        result = run_all_numeric_tests(df, gender_a="male", gender_b="female")
        
        # Should only test columns that exist
        assert len(result) == 1
        assert result.iloc[0]["variable"] == "vader_compound"
    
    def test_run_all_numeric_tests_small_sample(self):
        """Test with very small sample (should return NaN p-values)."""
        df = pd.DataFrame({
            "gender": ["male", "female"],
            "vader_compound": [0.5, 0.7]
        })
        
        result = run_all_numeric_tests(df, gender_a="male", gender_b="female")
        
        # With only 1 sample per group, all p-values should be NaN
        assert pd.isna(result.iloc[0]["t_pvalue"])
        assert pd.isna(result.iloc[0]["mw_pvalue"])
        assert pd.isna(result.iloc[0]["ks_pvalue"])
    
    def test_run_all_numeric_tests_handles_nan_values(self):
        """Test that NaN values are handled properly."""
        df = pd.DataFrame({
            "gender": ["male", "male", "female", "female"],
            "vader_compound": [0.5, np.nan, 0.7, 0.8]
        })
        
        result = run_all_numeric_tests(df, gender_a="male", gender_b="female")
        
        # Should drop NaN and compute on remaining values
        vader_row = result[result.variable == "vader_compound"].iloc[0]
        assert vader_row["n_a"] == 1  # Only 1 non-NaN male value
        assert vader_row["n_b"] == 2  # 2 non-NaN female values


class TestChiSquareTest:
    """Test chi-square test for RoBERTa labels."""
    
    def test_run_chi_square_test_basic(self):
        """Test chi-square on label distribution."""
        df = pd.DataFrame({
            "gender": ["male"] * 150 + ["female"] * 100,
            "roberta_label_str": (
                ["positive"] * 70 + ["neutral"] * 50 + ["negative"] * 30 +  # Male
                ["positive"] * 65 + ["neutral"] * 25 + ["negative"] * 10    # Female
            )
        })
        
        result = run_chi_square_test(df, gender_a="male", gender_b="female")
        
        assert isinstance(result, pd.DataFrame)
        assert "chi2" in result.columns
        assert "pvalue" in result.columns
        assert "dof" in result.columns
        
        # Should detect difference in distributions
        assert result.iloc[0]["pvalue"] < 0.05
        assert result.iloc[0]["chi2"] > 0
        assert result.iloc[0]["dof"] == 2  # (2 genders - 1) * (3 labels - 1)
    
    def test_run_chi_square_test_similar_distributions(self):
        """Test chi-square when distributions are similar."""
        df = pd.DataFrame({
            "gender": ["male"] * 100 + ["female"] * 100,
            "roberta_label_str": (
                ["positive"] * 33 + ["neutral"] * 34 + ["negative"] * 33 +
                ["positive"] * 33 + ["neutral"] * 34 + ["negative"] * 33
            )
        })
        
        result = run_chi_square_test(df, gender_a="male", gender_b="female")
        
        # Similar distributions should have high p-value
        assert result.iloc[0]["pvalue"] > 0.05
    
    def test_run_chi_square_test_missing_column(self):
        """Test handling when roberta_label_str column is missing."""
        df = pd.DataFrame({
            "gender": ["male", "female"],
            "vader_compound": [0.5, 0.7]
        })
        
        result = run_chi_square_test(df, gender_a="male", gender_b="female")
        
        # Should return DataFrame with NaN values
        assert isinstance(result, pd.DataFrame)
        assert pd.isna(result.iloc[0]["chi2"])
        assert pd.isna(result.iloc[0]["pvalue"])
        assert pd.isna(result.iloc[0]["dof"])


class TestSummaryStatistics:
    """Test summary statistics functions."""
    
    @pytest.fixture
    def sentiment_data(self):
        """Create sample sentiment data."""
        return pd.DataFrame({
            "gender": ["male"] * 50 + ["female"] * 50 + ["unknown"] * 20,
            "vader_neg": np.random.uniform(0, 0.2, 120),
            "vader_neu": np.random.uniform(0.5, 0.7, 120),
            "vader_pos": np.random.uniform(0.2, 0.4, 120),
            "vader_compound": np.random.uniform(-0.5, 0.8, 120),
            "textblob_polarity": np.random.uniform(-0.3, 0.5, 120),
            "textblob_subjectivity": np.random.uniform(0.2, 0.6, 120)
        })
    
    def test_compute_sentiment_summary_basic(self, sentiment_data):
        """Test sentiment summary computation."""
        result = compute_sentiment_summary(sentiment_data)
        
        assert isinstance(result, pd.DataFrame)
        
        # Should have multi-level columns (mean, std, count)
        assert "mean" in result.columns.get_level_values(1)
        assert "std" in result.columns.get_level_values(1)
        assert "count" in result.columns.get_level_values(1)
        
        # Should have all sentiment columns
        expected_cols = [
            "vader_neg", "vader_neu", "vader_pos", "vader_compound",
            "textblob_polarity", "textblob_subjectivity"
        ]
        for col in expected_cols:
            assert col in result.columns.get_level_values(0)
        
        # Should have all gender groups
        assert "male" in result.index
        assert "female" in result.index
        assert "unknown" in result.index
    
    def test_compute_sentiment_summary_values(self, sentiment_data):
        """Test that summary values are reasonable."""
        result = compute_sentiment_summary(sentiment_data)
        
        # Counts should match input
        assert result.loc["male", ("vader_compound", "count")] == 50
        assert result.loc["female", ("vader_compound", "count")] == 50
        
        # Means should be within reasonable range
        male_mean = result.loc["male", ("vader_compound", "mean")]
        assert -1 <= male_mean <= 1
    
    def test_compute_sentiment_summary_missing_gender(self):
        """Test handling when gender column is missing."""
        df = pd.DataFrame({
            "vader_compound": [0.5, 0.7]
        })
        
        result = compute_sentiment_summary(df)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_compute_sentiment_summary_partial_columns(self):
        """Test with only some sentiment columns present."""
        df = pd.DataFrame({
            "gender": ["male", "female"],
            "vader_compound": [0.5, 0.7]
        })
        
        result = compute_sentiment_summary(df)
        
        # Should only summarize columns that exist
        assert "vader_compound" in result.columns.get_level_values(0)
        assert "textblob_polarity" not in result.columns.get_level_values(0)


class TestRobertaDistribution:
    """Test RoBERTa label distribution summary."""
    
    def test_compute_roberta_distribution_basic(self):
        """Test RoBERTa distribution computation."""
        df = pd.DataFrame({
            "gender": ["male"] * 100 + ["female"] * 100,
            "roberta_label_str": (
                ["positive"] * 50 + ["neutral"] * 30 + ["negative"] * 20 +
                ["positive"] * 60 + ["neutral"] * 25 + ["negative"] * 15
            )
        })
        
        result = compute_roberta_distribution(df)
        
        assert isinstance(result, pd.DataFrame)
        
        # Should have gender as index
        assert "male" in result.index
        assert "female" in result.index
        
        # Should have label columns
        assert "positive" in result.columns
        assert "neutral" in result.columns
        assert "negative" in result.columns
        
        # Values should be proportions (sum to 1 per row)
        male_sum = result.loc["male"].sum()
        assert abs(male_sum - 1.0) < 0.01  # Allow small float error
    
    def test_compute_roberta_distribution_proportions(self):
        """Test that proportions are calculated correctly."""
        df = pd.DataFrame({
            "gender": ["male"] * 100,
            "roberta_label_str": ["positive"] * 50 + ["negative"] * 50
        })
        
        result = compute_roberta_distribution(df)
        
        # Should be 50% positive, 50% negative
        assert abs(result.loc["male", "positive"] - 0.5) < 0.01
        assert abs(result.loc["male", "negative"] - 0.5) < 0.01
    
    def test_compute_roberta_distribution_missing_columns(self):
        """Test handling when required columns are missing."""
        df = pd.DataFrame({
            "gender": ["male", "female"],
            "vader_compound": [0.5, 0.7]
        })
        
        result = compute_roberta_distribution(df)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestPronounSummary:
    """Test pronoun statistics summary."""
    
    def test_compute_pronoun_summary_basic(self):
        """Test pronoun summary computation."""
        df = pd.DataFrame({
            "gender": ["male"] * 50 + ["female"] * 50,
            "male_pronoun_count": np.concatenate([
                np.random.poisson(20, 50),
                np.random.poisson(2, 50)
            ]),
            "female_pronoun_count": np.concatenate([
                np.random.poisson(1, 50),
                np.random.poisson(22, 50)
            ]),
            "pronoun_male_minus_female": np.random.randint(-30, 30, 100)
        })
        
        result = compute_pronoun_summary(df)
        
        assert isinstance(result, pd.DataFrame)
        
        # Should have multi-level columns
        assert "mean" in result.columns.get_level_values(1)
        assert "std" in result.columns.get_level_values(1)
        assert "count" in result.columns.get_level_values(1)
        
        # Should have pronoun columns
        assert "male_pronoun_count" in result.columns.get_level_values(0)
        assert "female_pronoun_count" in result.columns.get_level_values(0)
    
    def test_compute_pronoun_summary_values(self):
        """Test that pronoun summary values are reasonable."""
        df = pd.DataFrame({
            "gender": ["male"] * 100,
            "male_pronoun_count": [20] * 100,
            "female_pronoun_count": [2] * 100
        })
        
        result = compute_pronoun_summary(df)
        
        # Mean should match input
        assert result.loc["male", ("male_pronoun_count", "mean")] == 20
        assert result.loc["male", ("female_pronoun_count", "mean")] == 2
        
        # Count should be 100
        assert result.loc["male", ("male_pronoun_count", "count")] == 100
    
    def test_compute_pronoun_summary_missing_columns(self):
        """Test handling when pronoun columns are missing."""
        df = pd.DataFrame({
            "gender": ["male", "female"],
            "vader_compound": [0.5, 0.7]
        })
        
        result = compute_pronoun_summary(df)
        
        # Should return empty DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestAnalysisIntegration:
    """Integration tests for complete analysis workflow."""
    
    def test_full_analysis_pipeline(self):
        """Test complete statistical analysis workflow."""
        # Create realistic dataset
        np.random.seed(42)
        df = pd.DataFrame({
            "gender": ["male"] * 100 + ["female"] * 100,
            "vader_compound": np.concatenate([
                np.random.normal(0.64, 0.2, 100),
                np.random.normal(0.77, 0.2, 100)
            ]),
            "textblob_polarity": np.concatenate([
                np.random.normal(0.078, 0.1, 100),
                np.random.normal(0.103, 0.1, 100)
            ]),
            "textblob_subjectivity": np.concatenate([
                np.random.normal(0.322, 0.05, 100),
                np.random.normal(0.338, 0.05, 100)
            ]),
            "male_pronoun_count": np.concatenate([
                np.random.poisson(20, 100),
                np.random.poisson(2, 100)
            ]),
            "roberta_label_str": (
                ["positive"] * 60 + ["neutral"] * 25 + ["negative"] * 15 +
                ["positive"] * 70 + ["neutral"] * 20 + ["negative"] * 10
            )
        })
        
        # Run all analyses
        numeric_results = run_all_numeric_tests(df, "male", "female")
        chi2_results = run_chi_square_test(df, "male", "female")
        sentiment_summary = compute_sentiment_summary(df)
        roberta_dist = compute_roberta_distribution(df)
        
        # Verify all ran successfully
        assert len(numeric_results) > 0
        assert len(chi2_results) > 0
        assert len(sentiment_summary) > 0
        assert len(roberta_dist) > 0
        
        # Verify significant differences are detected
        vader_test = numeric_results[numeric_results.variable == "vader_compound"].iloc[0]
        assert vader_test["t_pvalue"] < 0.05