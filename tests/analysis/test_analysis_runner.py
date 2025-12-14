# tests/analysis/test_analysis_runner.py

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock
from dsan_5400_group3.analysis.analysis_runner import AnalysisRunner

@pytest.fixture
def mock_deps(mocker):
    """Mocks external functions used by AnalysisRunner."""
    return {
        "summaries": mocker.patch("dsan_5400_group3.analysis.analysis_runner.compute_sentiment_summary"),
        "roberta_dist": mocker.patch("dsan_5400_group3.analysis.analysis_runner.compute_roberta_distribution"),
        "pronoun_summary": mocker.patch("dsan_5400_group3.analysis.analysis_runner.compute_pronoun_summary"),
        "numeric_tests": mocker.patch("dsan_5400_group3.analysis.analysis_runner.run_all_numeric_tests"),
        "chi_square": mocker.patch("dsan_5400_group3.analysis.analysis_runner.run_chi_square_test"),
        "save_func": mocker.patch("pandas.DataFrame.to_csv"), # Mock the save function
    }


@pytest.fixture
def mock_input_csv(tmp_path):
    """Creates a mock input CSV file for the runner to load."""
    data = {
        "gender": ["male", "female", "male"],
        "vader_compound": [0.5, 0.7, 0.6],
        "roberta_label_str": ["positive", "negative", "neutral"],
        "male_pronoun_count": [10, 1, 12],
    }
    df = pd.DataFrame(data)
    input_file = tmp_path / "biographies_with_sentiment.csv"
    df.to_csv(input_file, index=False)
    return input_file

class TestAnalysisRunner:

    def test_runner_initialization(self, mock_input_csv, tmp_path):
        """Test that the runner initializes correctly with arguments."""
        runner = AnalysisRunner(
            input_csv=mock_input_csv,
            output_dir=tmp_path / "results",
            sample_n=100,
            gender_a="group_a",
            gender_b="group_b",
        )
        assert runner.input_csv == mock_input_csv
        assert runner.gender_a == "group_a"
        assert runner.sample_n == 100

    def test_runner_runs_all_pipelines(self, mock_input_csv, tmp_path, mock_deps):
        """
        Tests that the run() method calls all expected analysis/summary functions.
        """
        output_dir = tmp_path / "results"
        output_dir.mkdir()
        
        # 1. Instantiate the Runner
        runner = AnalysisRunner(
            input_csv=mock_input_csv,
            output_dir=output_dir,
            sample_n=None,
            gender_a="male",
            gender_b="female",
        )
        
        # 2. Run the main method
        runner.run()

        # 3. Assertions: Check that the mocked analysis functions were called
        mock_deps["summaries"].assert_called_once()
        mock_deps["roberta_dist"].assert_called_once()
        mock_deps["pronoun_summary"].assert_called_once()
        mock_deps["numeric_tests"].assert_called_once()
        mock_deps["chi_square"].assert_called_once()
        
        # check if the output directory was created
        assert output_dir.exists()