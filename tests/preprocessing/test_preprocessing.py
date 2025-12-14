"""
Tests for preprocessing module.

Tests cover:
- Text cleaning
- Pronoun-based gender annotation
- Data loading
- ChunkedCleanerPipeline
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile

from dsan_5400_group3.preprocessing.cleaner import (
    normalize_unicode,
    remove_control_characters,
    remove_references_and_tags,
    remove_templates_and_tables,
    remove_maintenance_phrases,
    clean_text,
    ChunkedCleanerPipeline
)
from dsan_5400_group3.preprocessing.pronoun_annotator import (
    add_pronoun_gender_chunk,
    PronounGenderAnnotator,
    MALE_PRONOUNS,
    FEMALE_PRONOUNS
)
from dsan_5400_group3.preprocessing.loader import RawJSONLLoader


class TestCleaner:
    """Test text cleaning functions."""
    
    def test_normalize_unicode(self):
        """Test unicode normalization."""
        text = "café"  # Contains é
        result = normalize_unicode(text)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_remove_control_characters(self):
        """Test control character removal."""
        text = "Hello\x00World\x1F"
        cleaned = remove_control_characters(text)
        assert "\x00" not in cleaned
        assert "\x1F" not in cleaned
        assert "Hello" in cleaned
        assert "World" in cleaned
    
    def test_remove_references_and_tags_citations(self):
        """Test that Wikipedia citation markers are removed."""
        text = "Albert Einstein[1] was born in 1879[2]."
        cleaned = remove_references_and_tags(text)
        assert "[1]" not in cleaned
        assert "[2]" not in cleaned
        assert "Albert Einstein" in cleaned
    
    def test_remove_references_and_tags_html(self):
        """Test that HTML tags are removed."""
        text = "<p>Marie Curie</p> was a <strong>scientist</strong>."
        cleaned = remove_references_and_tags(text)
        assert "<p>" not in cleaned
        assert "</p>" not in cleaned
        assert "<strong>" not in cleaned
        assert "Marie Curie" in cleaned
    
    def test_remove_references_and_tags_ref_tags(self):
        """Test that <ref> tags are removed."""
        text = 'Einstein<ref>Source here</ref> was brilliant.'
        cleaned = remove_references_and_tags(text)
        assert "<ref>" not in cleaned
        assert "</ref>" not in cleaned
        assert "Source here" not in cleaned
    
    def test_remove_templates_and_tables(self):
        """Test that Wikipedia templates are removed."""
        text = "{{Infobox person}} Ada Lovelace {{cite book}}"
        cleaned = remove_templates_and_tables(text)
        assert "{{" not in cleaned
        assert "}}" not in cleaned
        assert "Ada Lovelace" in cleaned
    
    def test_remove_templates_and_tables_categories(self):
        """Test that categories are removed."""
        text = "Text here [[Category:Scientists]] more text"
        cleaned = remove_templates_and_tables(text)
        assert "[[Category:" not in cleaned
        assert "Scientists]]" not in cleaned
    
    def test_remove_templates_and_tables_images(self):
        """Test that image links are removed."""
        text = "Text [[File:Photo.jpg]] and [[Image:Pic.png]] more"
        cleaned = remove_templates_and_tables(text)
        assert "[[File:" not in cleaned
        assert "[[Image:" not in cleaned
    
    def test_remove_maintenance_phrases(self):
        """Test removal of Wikipedia maintenance phrases."""
        text = "this article has multiple issues. Some real content here."
        cleaned = remove_maintenance_phrases(text)
        assert "multiple issues" not in cleaned
    
    def test_clean_text_full_pipeline(self):
        """Test complete cleaning pipeline."""
        text = "<p>He[1] was a {{notable}} scientist[edit].</p>"
        cleaned = clean_text(text)
        
        # Should be lowercased
        assert cleaned == cleaned.lower()
        
        # Should remove all markup
        assert "<p>" not in cleaned
        assert "[1]" not in cleaned
        assert "{{" not in cleaned
        assert "[edit]" not in cleaned
        
        # Should preserve content
        assert "he" in cleaned
        assert "scientist" in cleaned
    
    def test_clean_text_normalizes_whitespace(self):
        """Test that excessive whitespace is collapsed."""
        text = "Isaac    Newton     was     born"
        cleaned = clean_text(text)
        assert "    " not in cleaned
        # Should have single spaces only
        assert "  " not in cleaned
    
    def test_clean_text_lowercases(self):
        """Test that text is lowercased."""
        text = "CHARLES DARWIN"
        cleaned = clean_text(text)
        assert cleaned == "charles darwin"
    
    def test_clean_text_handles_none(self):
        """Test that None input is handled gracefully."""
        result = clean_text(None)
        assert result == ""
    
    def test_clean_text_handles_empty_string(self):
        """Test that empty string is handled."""
        result = clean_text("")
        assert result == ""
    
    def test_clean_text_handles_non_string(self):
        """Test that non-string input is converted."""
        result = clean_text(12345)
        assert isinstance(result, str)
        assert "12345" in result


class TestPronounAnnotator:
    """Test pronoun counting and gender assignment."""
    
    def test_add_pronoun_gender_chunk_male(self):
        """Test gender assignment for male biography."""
        chunk = pd.DataFrame({
            "text": ["He was a scientist. His work was important. Him and his colleagues."]
        })
        
        result = add_pronoun_gender_chunk(chunk)
        
        assert "male_pronoun_count" in result.columns
        assert "female_pronoun_count" in result.columns
        assert "gender" in result.columns
        
        assert result.iloc[0]["male_pronoun_count"] >= 3
        assert result.iloc[0]["female_pronoun_count"] == 0
        assert result.iloc[0]["gender"] == "male"
    
    def test_add_pronoun_gender_chunk_female(self):
        """Test gender assignment for female biography."""
        chunk = pd.DataFrame({
            "text": ["She was a scientist. Her work was important. She and her colleagues."]
        })
        
        result = add_pronoun_gender_chunk(chunk)
        
        assert result.iloc[0]["female_pronoun_count"] >= 3
        assert result.iloc[0]["male_pronoun_count"] == 0
        assert result.iloc[0]["gender"] == "female"
    
    def test_add_pronoun_gender_chunk_unknown(self):
        """Test gender assignment when counts are equal."""
        chunk = pd.DataFrame({
            "text": ["The scientist was born in 1900."]
        })
        
        result = add_pronoun_gender_chunk(chunk)
        
        # Equal counts (both 0) should result in unknown
        assert result.iloc[0]["gender"] == "unknown"
    
    def test_add_pronoun_gender_chunk_mixed(self):
        """Test with mixed gender pronouns."""
        chunk = pd.DataFrame({
            "text": ["He and she worked together."]
        })
        
        result = add_pronoun_gender_chunk(chunk)
        
        # Should have both male and female pronouns
        assert result.iloc[0]["male_pronoun_count"] >= 1
        assert result.iloc[0]["female_pronoun_count"] >= 1
    
    def test_add_pronoun_gender_chunk_case_insensitive(self):
        """Test that pronoun counting is case-insensitive."""
        chunk = pd.DataFrame({
            "text": ["HE was important. HIS work was great."]
        })
        
        result = add_pronoun_gender_chunk(chunk)
        
        assert result.iloc[0]["male_pronoun_count"] >= 2
        assert result.iloc[0]["gender"] == "male"
    
    def test_add_pronoun_gender_chunk_handles_nan(self):
        """Test that NaN text is handled."""
        chunk = pd.DataFrame({
            "text": [None, "", "He was here"]
        })
        
        result = add_pronoun_gender_chunk(chunk)
        
        assert len(result) == 3
        assert result.iloc[0]["gender"] == "unknown"
        assert result.iloc[1]["gender"] == "unknown"
        assert result.iloc[2]["gender"] == "male"
    
    def test_add_pronoun_gender_chunk_multiple_rows(self):
        """Test processing multiple rows."""
        chunk = pd.DataFrame({
            "text": [
                "He was a scientist.",
                "She was an artist.",
                "They were writers."
            ]
        })
        
        result = add_pronoun_gender_chunk(chunk)
        
        assert len(result) == 3
        assert result.iloc[0]["gender"] == "male"
        assert result.iloc[1]["gender"] == "female"
        assert result.iloc[2]["gender"] == "unknown"
    
    def test_pronoun_constants(self):
        """Test that pronoun lists are defined."""
        assert "he" in MALE_PRONOUNS
        assert "him" in MALE_PRONOUNS
        assert "his" in MALE_PRONOUNS
        
        assert "she" in FEMALE_PRONOUNS
        assert "her" in FEMALE_PRONOUNS
        assert "hers" in FEMALE_PRONOUNS


class TestRawJSONLLoader:
    """Test JSONL loading functionality."""
    
    def test_load_all_single_file(self, tmp_path):
        """Test loading a single JSONL file."""
        import json
        
        # Create a single JSONL file
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps({"curid": "1", "text": "Article 1"}) + '\n')
            f.write(json.dumps({"curid": "2", "text": "Article 2"}) + '\n')
        
        loader = RawJSONLLoader(tmp_path)
        df = loader.load_all()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "curid" in df.columns
        assert "text" in df.columns
    
    def test_load_all_multiple_files(self, tmp_path):
        """Test loading multiple JSONL files."""
        import json
        
        # Create multiple JSONL files
        for i in range(3):
            jsonl_file = tmp_path / f"part_{i}.jsonl"
            with open(jsonl_file, 'w') as f:
                f.write(json.dumps({"curid": str(i), "text": f"Article {i}"}) + '\n')
        
        loader = RawJSONLLoader(tmp_path)
        df = loader.load_all()
        
        assert len(df) == 3
    
    def test_save_csv(self, tmp_path):
        """Test saving loaded data as CSV."""
        import json
        
        # Create JSONL file
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps({"curid": "1", "text": "Article 1"}) + '\n')
        
        loader = RawJSONLLoader(tmp_path)
        output_csv = tmp_path / "output.csv"
        
        result_path = loader.save_csv(output_csv)
        
        assert result_path.exists()
        
        # Verify CSV content
        df = pd.read_csv(result_path)
        assert len(df) == 1
        assert df.iloc[0]["curid"] == 1


class TestChunkedCleanerPipeline:
    """Test ChunkedCleanerPipeline."""
    
    def test_pipeline_initialization(self, tmp_path):
        """Test that pipeline initializes correctly."""
        raw_csv = tmp_path / "raw.csv"
        processed_dir = tmp_path / "processed"
        
        pipeline = ChunkedCleanerPipeline(raw_csv, processed_dir)
        
        assert pipeline.raw_csv == raw_csv
        assert pipeline.processed_dir == processed_dir
        assert pipeline.cleaned_csv == processed_dir / "biographies_clean.csv"
    
    def test_pipeline_run(self, tmp_path):
        """Test complete pipeline execution."""
        # Create sample raw CSV
        raw_csv = tmp_path / "raw.csv"
        df = pd.DataFrame({
            "text": [
                "<p>He[1] was a scientist{{template}}.</p>",
                "She was an artist[edit].",
                None  # Test NaN handling
            ],
            "missing": [False, False, False]
        })
        df.to_csv(raw_csv, index=False)
        
        processed_dir = tmp_path / "processed"
        
        pipeline = ChunkedCleanerPipeline(raw_csv, processed_dir, chunksize=2)
        result_path = pipeline.run()
        
        # Check output file exists
        assert result_path.exists()
        
        # Load and verify
        cleaned_df = pd.read_csv(result_path)
        
        # Should have cleaned text column
        assert "text_clean" in cleaned_df.columns
        
        # Should have length features
        assert "article_length_chars" in cleaned_df.columns
        assert "article_length_words" in cleaned_df.columns
        
        # Should have removed rows with empty text
        assert len(cleaned_df) >= 1
        
        # Text should be cleaned
        assert all(cleaned_df["text_clean"].str.len() > 0)


class TestPronounGenderAnnotatorIntegration:
    """Integration test for PronounGenderAnnotator."""
    
    def test_annotator_run(self, tmp_path):
        """Test complete annotation workflow."""
        # Create raw CSV
        raw_csv = tmp_path / "biographies_raw.csv"
        df = pd.DataFrame({
            "text": [
                "He was a scientist. His work was important.",
                "She was an artist. Her paintings were famous.",
                "The writer published many books."
            ]
        })
        df.to_csv(raw_csv, index=False)
        
        # Run annotator
        annotator = PronounGenderAnnotator(raw_csv)
        annotator.run(chunksize=2)
        
        # Load result
        result_df = pd.read_csv(raw_csv)
        
        # Should have gender columns
        assert "male_pronoun_count" in result_df.columns
        assert "female_pronoun_count" in result_df.columns
        assert "gender" in result_df.columns
        
        # Verify gender assignments
        assert result_df.iloc[0]["gender"] == "male"
        assert result_df.iloc[1]["gender"] == "female"
        assert result_df.iloc[2]["gender"] == "unknown"


class TestPreprocessingIntegration:
    """End-to-end integration tests."""
    
    def test_full_preprocessing_workflow(self, tmp_path):
        """Test complete preprocessing from raw to clean."""
        # Step 1: Create raw data
        import json
        jsonl_file = tmp_path / "data.jsonl"
        with open(jsonl_file, 'w') as f:
            f.write(json.dumps({
                "curid": "1",
                "text": "<b>He</b> was a scientist[1]."
            }) + '\n')
            f.write(json.dumps({
                "curid": "2", 
                "text": "She was an artist."
            }) + '\n')
        
        # Step 2: Load JSONL
        loader = RawJSONLLoader(tmp_path)
        raw_csv = tmp_path / "raw.csv"
        loader.save_csv(raw_csv)
        
        # Step 3: Add gender
        annotator = PronounGenderAnnotator(raw_csv)
        annotator.run(chunksize=10)
        
        # Step 4: Clean text
        processed_dir = tmp_path / "processed"
        cleaner = ChunkedCleanerPipeline(raw_csv, processed_dir, chunksize=10)
        cleaned_csv = cleaner.run()
        
        # Verify final output
        final_df = pd.read_csv(cleaned_csv)
        
        assert "text_clean" in final_df.columns
        assert "gender" in final_df.columns
        assert len(final_df) == 2
        
        # Text should be cleaned
        assert "<b>" not in final_df.iloc[0]["text_clean"]
        assert "[1]" not in final_df.iloc[0]["text_clean"]