import pytest
from pathlib import Path
from unittest.mock import MagicMock
from dsan_5400_group3.scraping.scraping_runner import WikiScraper
from multiprocessing import Process


@pytest.fixture
def mock_url_file(tmp_path):
    """Creates a mock input URL list file."""
    url_file = tmp_path / "url_list.txt"
    url_file.write_text("http://mock.com/page?curid=101\n")
    return str(url_file)


@pytest.fixture
def mock_scraper_deps(mocker):
    """Mocks list splitting and file loading, and replaces Process with a safe mock."""
    
    # --- STEP 1: Replace Process with a mock that won't try to pickle ---
    mock_process_instance = MagicMock()
    mock_process_instance.start.return_value = None
    mock_process_instance.join.return_value = None
    
    mock_process_class = mocker.patch(
        "dsan_5400_group3.scraping.scraping_runner.Process", # Patch Process where the runner imports it
        return_value=mock_process_instance
    )
    
    # --- STEP 2: Mock the helper functions ---
    
    # MOCK URL LOADING (used by run)
    mock_loader = mocker.patch(
        "dsan_5400_group3.scraping.scraping_runner.WikiScraper.load_urls",
        return_value=['id_101', 'id_102', 'id_103'] # Return mock IDs
    )
    
    mock_splitter = mocker.patch(
        "dsan_5400_group3.scraping.scraping_runner.split_into_chunks",
        return_value=[['id_101'], ['id_102'], ['id_103'], []]
    )
    
    # Mock Fetcher
    mock_fetcher = mocker.patch("dsan_5400_group3.scraping.fetcher.fetch_html_page")
    
    return {
        "splitter": mock_splitter,
        "process": mock_process_class, # Pass the mock class
        "process_instance": mock_process_instance, # Pass the mock instance for call counts
        "loader": mock_loader,
        "fetcher": mock_fetcher
    }


def test_scraper_run_initialization_and_orchestration(tmp_path, mock_url_file, mock_scraper_deps):
    """Test that WikiScraper coordinates all steps without triggering pickling."""
    outdir = tmp_path / "scraper_output"
    workers = 4 
    mock_chunks = mock_scraper_deps["splitter"].return_value
    
    scraper = WikiScraper(workers=workers, outdir=outdir)

    # 1. Run the scraper
    scraper.run(mock_url_file)

    # 2. Assertions: 
    
    # A. URL loading and Splitting assertions (as before, they should pass)
    mock_scraper_deps["loader"].assert_called_once_with(mock_url_file)
    mock_scraper_deps["splitter"].assert_called_once_with(
        mock_scraper_deps["loader"].return_value, 
        workers
    )

    # C. Assert Process was instantiated for each chunk
    mock_process_class = mock_scraper_deps["process"]
    assert mock_process_class.call_count == len(mock_chunks)
    
    # D. Assert correct arguments were passed to Process (Target and args[1])
    for i, chunk in enumerate(mock_chunks):
        call_kwargs = mock_process_class.call_args_list[i][1] 
        
        # Target must be the actual method bound to the scraper instance
        assert call_kwargs['target'] == scraper._worker 
        assert call_kwargs['args'][1] == chunk
        
    # E. Assert the start and join methods were called on the mock instance
    mock_process_instance = mock_scraper_deps["process_instance"]
    assert mock_process_instance.start.call_count == len(mock_chunks)
    assert mock_process_instance.join.call_count == len(mock_chunks)

    print("\n\n*** SUCCESS: Scraping orchestration test passed by safely mocking Process. ***")