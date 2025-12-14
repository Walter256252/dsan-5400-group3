import pytest
import pandas as pd 
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock 
from dsan_5400_group3.sentiment.sentiment_runner import SentimentRunner



@pytest.fixture
def mock_input_data(tmp_path):
    """
    Creates the mock input CSV file on disk for existence checks.
    Returns the file path and the mock data used for pd.read_csv.
    """
    data = {
        "text_clean": ["Good text here.", "Bad text here."],
        "article_length_words": [5, 5]
    }
    df = pd.DataFrame(data)
    
    # Create the input file in the temp directory that the *first* test will check for existence
    input_file = tmp_path / "input_for_test.csv"
    df.to_csv(input_file, index=False)
    
    # Return the path to the temp file, and the DataFrame content
    return input_file, df


@pytest.fixture
def mock_sentiment_deps(mocker, mock_input_data):
    """Mocks external I/O and all sentiment scoring modules."""
    
    input_file_path, df_data = mock_input_data
    
    # 1. MOCK FILE READING: 
    mock_read_csv = mocker.patch(
        "dsan_5400_group3.sentiment.sentiment_runner.pd.read_csv", 
        return_value=df_data.copy()
    )
    
    # 2. MOCK GITHUB DOWNLOAD: (Fix for attribute error + logic)
    mock_gdrive_method = mocker.patch(
        "dsan_5400_group3.sentiment.sentiment_runner.SentimentRunner.download_from_drive"
    )
    
    # FIX 2: Set the mock download function's side_effect to physically create the expected file 
    # This prevents FileNotFoundError in tests that rely on the downloaded file existing.
    def mock_download_and_create_file(drive_url, out_path):
        """Simulates gdown.download by ensuring the output path exists."""
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # We must create the file so Path.exists() check passes in load_data()
        out_path.write_text("curid,text_clean\n1,data") 
        
    mock_gdrive_method.side_effect = mock_download_and_create_file


    # 3. MOCK ROBERTA: 
    mock_roberta_class = mocker.patch("dsan_5400_group3.sentiment.sentiment_runner.RobertaSentiment")
    mock_roberta_instance = mock_roberta_class.return_value
    mock_roberta_instance.add_roberta_sentiment.return_value = df_data.copy()

    return {
        "read_csv": mock_read_csv,
        "vader": mocker.patch("dsan_5400_group3.sentiment.sentiment_runner.add_vader_sentiment"),
        "textblob": mocker.patch("dsan_5400_group3.sentiment.sentiment_runner.add_textblob_sentiment"),
        "roberta_class": mock_roberta_class,
        "gdrive_downloader": mock_gdrive_method # Use the corrected name
    }


class TestSentimentRunner:

    def test_runner_calls_all_sentiment_methods(self, tmp_path, mock_sentiment_deps, mock_input_data):
        """Test that the run() method correctly orchestrates all three sentiment analyzers."""
        
        input_csv, _ = mock_input_data # Use the path created by the fixture
        output_csv = tmp_path / "output.csv"
        
        runner = SentimentRunner(
            input_csv=input_csv,
            output_csv=output_csv,
            batch_size=32,
            max_length=256,
        )
        
        # Run the pipeline
        runner.run()
        
        # Assert input file was read with the correct path
        mock_sentiment_deps["read_csv"].assert_called_once_with(
            input_csv
        )
        
        # Assert VADER, TextBlob, RoBERTa were called
        mock_sentiment_deps["vader"].assert_called_once()
        mock_sentiment_deps["textblob"].assert_called_once()
        mock_sentiment_deps["roberta_class"].return_value.add_roberta_sentiment.assert_called_once()

    def test_runner_handles_gdrive_download(self, tmp_path, mock_sentiment_deps):
        """Test that the runner uses the Google Drive downloader when a URL is provided."""
        
        drive_url = "http://mock-drive-link"
        
        # FIX 3: We MUST define the input path relative to the test runner 
        # so that the mock download can be verified and the final read can be mocked.
        # This is the expected path used by the runner (data/downloads/biographies_clean.csv).
        input_csv_path = Path("data/downloads/biographies_clean.csv")
        output_csv = tmp_path / "out.csv"
        
        # 1. Setup the Runner
        runner = SentimentRunner(
            input_csv=input_csv_path, # Path points to the conventional (mocked) location
            output_csv=output_csv,
            drive_url=drive_url,
        )
        
        # 2. Run the pipeline (this path should trigger the download_from_drive method)
        # The side_effect set in the fixture will now ensure input_csv_path exists before pd.read_csv is called
        runner.run()
        
        # 3. Assertions
        
        # Assert the downloader was called with the correct parameters
        mock_sentiment_deps["gdrive_downloader"].assert_called_once_with(
            drive_url, 
            input_csv_path # Asserts the correct path was passed
        )
        
        # Assert read_csv was called with the local path after the "download"
        mock_sentiment_deps["read_csv"].assert_called_once_with(
            input_csv_path
        )