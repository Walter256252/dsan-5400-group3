import pytest
from pathlib import Path
from unittest.mock import MagicMock
from dsan_5400_group3.preprocessing.preprocessing_runner import Preprocessor 


@pytest.fixture
def mock_preprocessing_deps(mocker, tmp_path):
    """Mocks the core components run by the Preprocessor."""
    # Mock the internal logic components
    mock_loader = mocker.patch("dsan_5400_group3.preprocessing.preprocessing_runner.RawJSONLLoader")
    mock_annotator = mocker.patch("dsan_5400_group3.preprocessing.preprocessing_runner.PronounGenderAnnotator")
    mock_cleaner_pipeline = mocker.patch("dsan_5400_group3.preprocessing.preprocessing_runner.ChunkedCleanerPipeline")
    
    
    return {
        "loader": mock_loader,
        "annotator": mock_annotator,
        "cleaner_pipeline": mock_cleaner_pipeline,
    }


def test_preprocessor_run_all_calls_steps_in_order(tmp_path, mock_preprocessing_deps):
    """Test that the full preprocessing pipeline executes all steps correctly."""
    
    # We use tmp_path as the mock project root.
    preprocessor = Preprocessor(project_root=tmp_path)
    
    preprocessor.run_all()
    
    # --- 1. Assert RawJSONLLoader was instantiated and save_csv was called ---
    # The loader is instantiated with the raw directory attribute: self.raw_dir
    mock_preprocessing_deps["loader"].assert_called_once_with(
        preprocessor.raw_dir
    )
    # The loader's save_csv method is called with the raw CSV file attribute: self.raw_csv
    mock_preprocessing_deps["loader"].return_value.save_csv.assert_called_once_with(
        preprocessor.raw_csv #
    )
    
    # --- 2. Assert PronounGenderAnnotator was instantiated and run ---
    # Annotator is instantiated with the raw CSV file attribute: self.raw_csv
    mock_preprocessing_deps["annotator"].assert_called_once_with(preprocessor.raw_csv)
    mock_preprocessing_deps["annotator"].return_value.run.assert_called_once()
    
    # --- 3. Assert ChunkedCleanerPipeline was instantiated and run ---
    # Cleaner is instantiated with self.raw_csv and self.processed_dir attributes.
    mock_preprocessing_deps["cleaner_pipeline"].assert_called_once_with(
        raw_csv=preprocessor.raw_csv,
        processed_dir=preprocessor.processed_dir
    )
    mock_preprocessing_deps["cleaner_pipeline"].return_value.run.assert_called_once()
