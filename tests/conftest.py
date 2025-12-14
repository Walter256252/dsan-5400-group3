"""
Pytest configuration and shared fixtures.

This file contains fixtures that are shared across all test files.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


@pytest.fixture
def sample_biographies():
    """
    Create a small sample dataset of biographies.
    
    Returns:
        DataFrame with realistic biography data
    """
    return pd.DataFrame({
        "text": [
            "He was born in 1879. His work revolutionized physics. He won the Nobel Prize.",
            "She was a pioneering scientist. Her research on radioactivity was groundbreaking.",
            "The author wrote many books. They lived in London for decades.",
            "He was a famous painter. His art style was revolutionary. He influenced many artists.",
            "She was an influential politician. Her policies changed the nation.",
        ],
        "curid": ["1", "2", "3", "4", "5"],
    })


@pytest.fixture
def sample_biographies_with_gender():
    """
    Create sample biographies with pre-assigned gender labels.
    
    Returns:
        DataFrame with biography text and gender labels
    """
    return pd.DataFrame({
        "text": [
            "He was a scientist. His discoveries were important.",
            "She was an artist. Her paintings were famous.",
            "The person was a writer.",
            "He was an inventor. His patents changed industry.",
            "She was a mathematician. Her theorems were elegant.",
        ],
        "gender": ["male", "female", "unknown", "male", "female"],
        "curid": ["1", "2", "3", "4", "5"],
    })


@pytest.fixture
def sample_sentiment_data():
    """
    Create sample data with sentiment scores.
    
    Returns:
        DataFrame with text, gender, and sentiment scores
    """
    np.random.seed(42)
    
    n_male = 100
    n_female = 100
    
    return pd.DataFrame({
        "text": ["Sample text"] * (n_male + n_female),
        "gender": ["male"] * n_male + ["female"] * n_female,
        "vader_compound": np.concatenate([
            np.random.normal(0.64, 0.2, n_male),
            np.random.normal(0.77, 0.2, n_female)
        ]),
        "vader_pos": np.concatenate([
            np.random.uniform(0.2, 0.4, n_male),
            np.random.uniform(0.25, 0.45, n_female)
        ]),
        "vader_neg": np.concatenate([
            np.random.uniform(0.05, 0.15, n_male),
            np.random.uniform(0.03, 0.12, n_female)
        ]),
        "vader_neu": np.concatenate([
            np.random.uniform(0.5, 0.7, n_male),
            np.random.uniform(0.45, 0.65, n_female)
        ]),
        "textblob_polarity": np.concatenate([
            np.random.normal(0.078, 0.1, n_male),
            np.random.normal(0.103, 0.1, n_female)
        ]),
        "textblob_subjectivity": np.concatenate([
            np.random.normal(0.322, 0.05, n_male),
            np.random.normal(0.338, 0.05, n_female)
        ]),
        "word_count": np.concatenate([
            np.random.normal(1478, 200, n_male).astype(int),
            np.random.normal(1549, 200, n_female).astype(int)
        ]),
        "char_count": np.concatenate([
            np.random.normal(8500, 1000, n_male).astype(int),
            np.random.normal(8900, 1000, n_female).astype(int)
        ])
    })


@pytest.fixture
def sample_roberta_data():
    """
    Create sample data with RoBERTa labels.
    
    Returns:
        DataFrame with gender and sentiment labels
    """
    return pd.DataFrame({
        "gender": ["male"] * 150 + ["female"] * 100,
        "roberta_label": (
            ["POSITIVE"] * 70 + ["NEUTRAL"] * 50 + ["NEGATIVE"] * 30 +
            ["POSITIVE"] * 65 + ["NEUTRAL"] * 25 + ["NEGATIVE"] * 10
        ),
        "roberta_score": np.random.uniform(0.7, 0.99, 250)
    })


@pytest.fixture
def temp_csv_file(tmp_path):
    """
    Create a temporary CSV file for testing.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Path to temporary CSV file
    """
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })
    
    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def temp_jsonl_file(tmp_path):
    """
    Create a temporary JSONL file for testing.
    
    Args:
        tmp_path: pytest temporary directory fixture
        
    Returns:
        Path to temporary JSONL file
    """
    import json
    
    data = [
        {"curid": "1", "text": "Article 1"},
        {"curid": "2", "text": "Article 2"},
    ]
    
    jsonl_path = tmp_path / "test_data.jsonl"
    with open(jsonl_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    return jsonl_path


@pytest.fixture
def mock_wikipedia_response():
    """
    Create a mock Wikipedia HTML response.
    
    Returns:
        Mock response object with HTML content
    """
    from unittest.mock import Mock
    
    mock = Mock()
    mock.status_code = 200
    mock.text = """
    <html>
    <body>
        <div class="mw-parser-output">
            <p><b>Albert Einstein</b> was a German-born theoretical physicist.</p>
            <p>He developed the theory of relativity.</p>
            <p>Einstein received the Nobel Prize in Physics in 1921.</p>
        </div>
    </body>
    </html>
    """
    return mock


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_model: marks tests that require ML models"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark tests that use real models as slow
        if "roberta" in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
            item.add_marker(pytest.mark.requires_model)
        
        # Mark integration tests
        if "integration" in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)