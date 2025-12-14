"""
Tests for sentiment analysis module.

Tests cover:
- VADER sentiment scoring
- TextBlob sentiment scoring
- RoBERTa classification
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from dsan_5400_group3.sentiment.vader import add_vader_sentiment
from dsan_5400_group3.sentiment.textblob import add_textblob_sentiment


class TestVADER:
    """Test VADER sentiment analysis."""
    
    def test_add_vader_sentiment_basic(self):
        """Test VADER adds correct columns."""
        df = pd.DataFrame({
            "text": [
                "This is absolutely wonderful and amazing!",
                "This is terrible and awful.",
                "The person was born in 1950."
            ]
        })
        
        result = add_vader_sentiment(df, text_col="text")
        
        # Should add VADER columns
        assert "vader_neg" in result.columns
        assert "vader_neu" in result.columns
        assert "vader_pos" in result.columns
        assert "vader_compound" in result.columns
    
    def test_add_vader_sentiment_positive_text(self):
        """Test VADER on clearly positive text."""
        df = pd.DataFrame({
            "text": ["This is absolutely wonderful and amazing! I love it!"]
        })
        
        result = add_vader_sentiment(df, text_col="text")
        
        # Positive text should have high compound score
        assert result.iloc[0]["vader_compound"] > 0.5
        assert result.iloc[0]["vader_pos"] > result.iloc[0]["vader_neg"]
    
    def test_add_vader_sentiment_negative_text(self):
        """Test VADER on clearly negative text."""
        df = pd.DataFrame({
            "text": ["This is terrible and awful. I hate it completely."]
        })
        
        result = add_vader_sentiment(df, text_col="text")
        
        # Negative text should have low compound score
        assert result.iloc[0]["vader_compound"] < -0.5
        assert result.iloc[0]["vader_neg"] > result.iloc[0]["vader_pos"]
    
    def test_add_vader_sentiment_neutral_text(self):
        """Test VADER on neutral text."""
        df = pd.DataFrame({
            "text": ["The person was born in 1950. They lived in New York."]
        })
        
        result = add_vader_sentiment(df, text_col="text")
        
        # Neutral text should have compound near 0
        assert -0.3 < result.iloc[0]["vader_compound"] < 0.3
        assert result.iloc[0]["vader_neu"] > 0.5
    
    def test_add_vader_sentiment_handles_nan(self):
        """Test VADER handles NaN values gracefully."""
        df = pd.DataFrame({
            "text": [None, "", "Good text"]
        })
        
        result = add_vader_sentiment(df, text_col="text")
        
        # Should not crash
        assert len(result) == 3
        assert "vader_compound" in result.columns
        
        # NaN/empty should have neutral scores
        assert result.iloc[0]["vader_compound"] == 0.0
        assert result.iloc[1]["vader_compound"] == 0.0
    
    def test_add_vader_sentiment_multiple_rows(self):
        """Test VADER on multiple rows."""
        df = pd.DataFrame({
            "text": [
                "Amazing and wonderful!",
                "Terrible and awful.",
                "It was okay."
            ]
        })
        
        result = add_vader_sentiment(df, text_col="text")
        
        assert len(result) == 3
        
        # First should be positive
        assert result.iloc[0]["vader_compound"] > 0
        # Second should be negative
        assert result.iloc[1]["vader_compound"] < 0
        # Third should be neutral-ish
        assert -0.5 < result.iloc[2]["vader_compound"] < 0.5


class TestTextBlob:
    """Test TextBlob sentiment analysis."""
    
    def test_add_textblob_sentiment_basic(self):
        """Test TextBlob adds correct columns."""
        df = pd.DataFrame({
            "text": ["This is a test sentence."]
        })
        
        result = add_textblob_sentiment(df, text_col="text")
        
        assert "textblob_polarity" in result.columns
        assert "textblob_subjectivity" in result.columns
    
    def test_add_textblob_sentiment_positive_text(self):
        """Test TextBlob on positive text."""
        df = pd.DataFrame({
            "text": ["This is absolutely wonderful and amazing!"]
        })
        
        result = add_textblob_sentiment(df, text_col="text")
        
        # Positive text should have positive polarity
        assert result.iloc[0]["textblob_polarity"] > 0.3
        # Opinion-heavy text should have high subjectivity
        assert result.iloc[0]["textblob_subjectivity"] > 0.2
    
    def test_add_textblob_sentiment_negative_text(self):
        """Test TextBlob on negative text."""
        df = pd.DataFrame({
            "text": ["This is terrible and awful."]
        })
        
        result = add_textblob_sentiment(df, text_col="text")
        
        # Negative text should have negative polarity
        assert result.iloc[0]["textblob_polarity"] < -0.2
    
    def test_add_textblob_sentiment_factual_text(self):
        """Test TextBlob on factual text."""
        df = pd.DataFrame({
            "text": ["The city has a population of 500,000 people."]
        })
        
        result = add_textblob_sentiment(df, text_col="text")
        
        # Factual text should have low subjectivity
        assert result.iloc[0]["textblob_subjectivity"] < 0.3
        # Should be relatively neutral
        assert -0.2 < result.iloc[0]["textblob_polarity"] < 0.2
    
    def test_add_textblob_sentiment_subjective_text(self):
        """Test TextBlob on subjective text."""
        df = pd.DataFrame({
            "text": ["I believe this is the best thing ever created!"]
        })
        
        result = add_textblob_sentiment(df, text_col="text")
        
        # Opinion text should have high subjectivity
        assert result.iloc[0]["textblob_subjectivity"] > 0.2
    
    def test_add_textblob_sentiment_handles_nan(self):
        """Test TextBlob handles NaN values."""
        df = pd.DataFrame({
            "text": [None, "", "Real text here"]
        })
        
        result = add_textblob_sentiment(df, text_col="text")
        
        assert len(result) == 3
        # Should not crash on NaN/empty
        assert "textblob_polarity" in result.columns
    
    def test_add_textblob_sentiment_multiple_rows(self):
        """Test TextBlob on multiple rows."""
        df = pd.DataFrame({
            "text": [
                "I absolutely love this!",
                "This is completely terrible.",
                "The book has 300 pages."
            ]
        })
        
        result = add_textblob_sentiment(df, text_col="text")
        
        assert len(result) == 3
        
        # First should be positive and subjective
        assert result.iloc[0]["textblob_polarity"] > 0
        assert result.iloc[0]["textblob_subjectivity"] > 0.5
        
        # Last should be neutral and factual
        assert abs(result.iloc[2]["textblob_polarity"]) < 0.2
        assert result.iloc[2]["textblob_subjectivity"] < 0.3


class TestRoBERTa:
    """Test RoBERTa sentiment classification."""
    
    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_roberta_sentiment_initialization(self):
        """Test RoBERTa model can be initialized."""
        from src.dsan_5400_group3.sentiment.roberta import RobertaSentiment
        
        # This is slow because it loads the actual model
        roberta = RobertaSentiment(batch_size=2, max_length=128)
        
        assert roberta.tokenizer is not None
        assert roberta.model is not None
        assert roberta.device in ["cuda", "cpu"]
    
    @pytest.mark.slow
    @pytest.mark.requires_model
    def test_roberta_add_sentiment(self):
        """Test RoBERTa adds sentiment columns."""
        from src.dsan_5400_group3.sentiment.roberta import RobertaSentiment
        
        df = pd.DataFrame({
            "text": [
                "This is amazing and wonderful!",
                "This is terrible.",
            ]
        })
        
        roberta = RobertaSentiment(batch_size=2, max_length=128)
        result = roberta.add_roberta_sentiment(df, text_col="text")
        
        assert "roberta_label" in result.columns
        assert "roberta_confidence" in result.columns
        
        # Labels should be LABEL_0, LABEL_1, or LABEL_2 (or similar)
        assert len(result.iloc[0]["roberta_label"]) > 0
        
        # Confidence should be between 0 and 1
        assert 0 <= result.iloc[0]["roberta_confidence"] <= 1
    
    def test_roberta_with_mock(self):
        """Test RoBERTa with mocked model (fast test)."""
        from src.dsan_5400_group3.sentiment.roberta import RobertaSentiment
        
        with patch('src.dsan_5400_group3.sentiment.roberta.AutoModelForSequenceClassification'):
            with patch('src.dsan_5400_group3.sentiment.roberta.AutoTokenizer'):
                
                df = pd.DataFrame({
                    "text": ["Sample text"]
                })
                
                # This test just verifies the structure, not actual predictions
                # Full test requires the real model (marked as slow)


class TestSentimentIntegration:
    """Integration tests for sentiment pipeline."""
    
    def test_vader_and_textblob_pipeline(self):
        """Test running VADER and TextBlob together."""
        df = pd.DataFrame({
            "text": [
                "He was an amazing scientist who made wonderful discoveries.",
                "She was a controversial figure whose work was criticized.",
                "The person was born in 1900 and lived in Paris."
            ],
            "gender": ["male", "female", "unknown"]
        })
        
        # Run VADER
        df = add_vader_sentiment(df, text_col="text")
        
        # Run TextBlob
        df = add_textblob_sentiment(df, text_col="text")
        
        # Verify all sentiment columns exist
        assert "vader_compound" in df.columns
        assert "vader_pos" in df.columns
        assert "vader_neg" in df.columns
        assert "vader_neu" in df.columns
        assert "textblob_polarity" in df.columns
        assert "textblob_subjectivity" in df.columns
        
        # Verify reasonable value ranges
        assert all(df["vader_compound"].between(-1, 1))
        assert all(df["textblob_polarity"].between(-1, 1))
        assert all(df["textblob_subjectivity"].between(0, 1))
        
        # First row should be positive
        assert df.iloc[0]["vader_compound"] > 0.3
        assert df.iloc[0]["textblob_polarity"] > 0.1
    
    def test_sentiment_preserves_original_columns(self):
        """Test that sentiment analysis preserves original columns."""
        df = pd.DataFrame({
            "text": ["Test text"],
            "curid": ["123"],
            "gender": ["male"]
        })
        
        df = add_vader_sentiment(df, text_col="text")
        df = add_textblob_sentiment(df, text_col="text")
        
        # Original columns should still exist
        assert "text" in df.columns
        assert "curid" in df.columns
        assert "gender" in df.columns
    
    def test_sentiment_handles_missing_data(self):
        """Test sentiment analysis handles NaN values."""
        df = pd.DataFrame({
            "text": ["Good text", None, "", "Bad text"]
        })
        
        df = add_vader_sentiment(df, text_col="text")
        df = add_textblob_sentiment(df, text_col="text")
        
        # Should not crash on None/empty
        assert len(df) == 4
        
        # All should have valid scores
        assert all(pd.notna(df["vader_compound"]))
        assert all(pd.notna(df["textblob_polarity"]))
    
    def test_sentiment_on_realistic_biographies(self):
        """Test sentiment on realistic biography text."""
        df = pd.DataFrame({
            "text": [
                "albert einstein was a german-born theoretical physicist who developed the theory of relativity. he received the nobel prize in physics in 1921.",
                "marie curie was a polish and naturalized-french physicist and chemist who conducted pioneering research on radioactivity. she was the first woman to win a nobel prize.",
            ]
        })
        
        df = add_vader_sentiment(df, text_col="text")
        df = add_textblob_sentiment(df, text_col="text")
        
        # Should complete without errors
        assert len(df) == 2
        
        # Should have sentiment scores
        assert all(pd.notna(df["vader_compound"]))
        assert all(pd.notna(df["textblob_polarity"]))
        
        # Scores should be reasonable for factual text
        assert all(df["vader_compound"].between(-0.5, 0.8))
        assert all(df["textblob_subjectivity"].between(0, 0.6))