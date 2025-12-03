# Gender Bias in Wikipedia Biographical Articles  
### DSAN 5400 – Group 3  
**Authors:** Laurent Julia Calac, Jiayuan Gong, Walter Hall, Jing Tan, Hung Tran  

---

# Project Overview

This project investigates **gender bias in Wikipedia biographical articles**, a widely studied issue in computational social science and NLP fairness. We examine whether the **sentiment**, **descriptive framing**, and **article structure** differ systematically between men’s and women’s biographies.

Using modern NLP models—including **transformer-based sentiment analysis**—we analyze more than **573,000** cleaned biographies.

---

# Research Question

> **Do Wikipedia biographies of men and women differ in sentiment, subjectivity, and descriptive patterns?**

We answer this using:
- Sentiment analysis (VADER, TextBlob, RoBERTa)
- Pronoun-based gender inference  
- Statistical testing across multiple distributions  
- Article structure metrics (length, subjectivity)

---

# Related Work

Prior research consistently identifies gender asymmetries in Wikipedia:

- **Wagner et al. (2015)** – Women’s biographies more often emphasize relationships and personal life.  
- **Graells-Garrido et al. (2015)** – Topic coverage differs by gender, with male biographies dominating central page networks.  
- **Field et al. (2021)** – Bias persists even accounting for profession and notability.  

Our project extends these findings using:
- Large-scale scraping  
- A modern transformer sentiment model  
- Rigorous statistical testing

---

# Data Access

To ensure full reproducibility without requiring re-scraping Wikipedia, we provide the complete cleaned and processed dataset via Google Drive:

**Dataset Folder (Google Drive):**  
**https://drive.google.com/drive/folders/1dvCqhO2KbiwJE9h4KIuqZIQJADgC_v0k**

This folder contains:

| File | Description | Download Link |
|------|-------------|----------------|
| `biographies_clean.csv` | Cleaned biography text (input for sentiment) | [Download](https://drive.google.com/file/d/1Z7Qm4MGp-lZI8j4k2px2wcHGjhrbpj6z/view?usp=sharing) |
| `biographies_with_sentiment.csv` | Sentiment-enhanced dataset (VADER, TextBlob, RoBERTa) | [Download](https://drive.google.com/file/d/1YWmOZT2o5BykMsA1Kd_YuDlWhpuK7CwU/view?usp=sharing) |
| `sentiment_summary_by_gender.csv` | Mean sentiment statistics | [Download](https://drive.google.com/file/d/19B6xoDCKnwqedErYpKEhbFmOBnx5T9xi/view?usp=sharing) |
| `pronoun_stats_by_gender.csv` | He/She pronoun-derived gender validation | [Download](https://drive.google.com/file/d/1J8Sor2dkd_w12iISowZAcfmpyo-VYCkk/view?usp=sharing) |
| `roberta_distribution_by_gender.csv` | Transformer label counts by gender | [Download](https://drive.google.com/file/d/1UnumR_5jljYa80lqlRYcCE_R1Yt9gmQZ/view?usp=sharing) |
| `stats_continuous_male_vs_female.csv` | t-test / U-test / KS-test results | [Download](https://drive.google.com/file/d/1v11r1PuEIOsL4WrieeEw7TBNoHjxAVXb/view?usp=sharing) |
| `stats_chi2_roberta_male_vs_female.csv` | Chi-square results for RoBERTa labels | [Download](https://drive.google.com/file/d/1a5ycWqwmN)_


Users may:

### Option A: Download manually  
Download any files you'd like directly from the Drive folder.

### Option B: Use `--drive-url` to auto-download  
Example:

```bash
poetry run python scripts/run_sentiment.py \
  --drive-url "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"

```
Replace FILE_ID with the exact file ID for the dataset you want to fetch.

---

## Methods

### **1. Data Acquisition**
Scraped Wikipedia biography pages using `?curid=###` links from `biography_urls.txt`.

Tools used:
- `scrape_data.py`
- Multiprocessing workers
- Automatic rate limiting
- HTML parsing with BeautifulSoup

---

### **2. Cleaning & Normalization**
`clean_data.py` performs:
- Unicode normalization  
- Removal of references, templates, HTML tags, categories  
- Lowercasing and whitespace collapsing  
- Removal of maintenance templates  
- Length statistics (characters, words)

Output: data/processed/biographies_clean.csv

---

### **3. Gender Assignment**
`add_pronoun_gender.py` counts:
- Male pronouns: **he / him / his**
- Female pronouns: **she / her / hers**

Articles with strong pronoun usage are assigned gender labels.

---

### **4. Sentiment Analysis**

Three sentiment models:

| Model | Type | Output |
|-------|-------|---------|
| **VADER** | Lexicon-based | polarity scores |
| **TextBlob** | Rule-based | polarity + subjectivity |
| **RoBERTa (CardiffNLP)** | Transformer | negative / neutral / positive |

Script:run_sentiment.py


---

### **5. Exploratory Analysis**
`run_analysis.py` computes:
- Sentiment averages by gender  
- Pronoun statistics  
- Article length distributions  
- Transformer label frequencies  

---

### **6. Statistical Testing**
Using `run_stats.py`, we compute:
- Student’s t-test  
- Mann–Whitney U test  
- Kolmogorov–Smirnov test  
- Chi-square test for RoBERTa labels  

Results stored in: results/sentiment/


---

## Reproducibility Instructions

### **1. Install Dependencies**
```bash
poetry install
```
### **2. Ensure Cleaned Data Exists**

Your working input **must be**:

- `data/processed/biographies_clean.csv`


To obtain it data:


## Download Cleaned Data from Google Drive

### Manual Download

- Download directly from Google Drive:
    - biographies_clean.csv
    - https://drive.google.com/file/d/1Z7Qm4MGp-lZI8j4k2px2wcHGjhrbpj6z/view?usp=sharing

After downloading, place the file here: data/processed/biographies_clean.csv

### Command-Line Download (Automatic)

- Your script already supports downloading via --drive-url.
Run the following command:

```bash 
poetry run python scripts/run_sentiment.py \
    --drive-url "https://drive.google.com/file/d/1Z7Qm4MGp-lZI8j4k2px2wcHGjhrbpj6z/view?usp=sharing"
```
This will:
- Download the CSV directly into
- data/processed/biographies_clean.csv
- Overwrite any existing file
- Continue with sentiment analysis

### Recommended Workflow

If you want maximum reproducibility + rapid setup:
- Download cleaned data from Drive
- Run sentiment + stats on 20k or full dataset
This avoids needing to re-run scraping or cleaning on ~573k articles.

---

## Running Sentiment Analysis & Stats

### Option A — 20k Sample (Fast, Recommended for Testing)

```bash
poetry run python scripts/run_sentiment.py \
  --sample-n 20000 \
  --output-path data/processed/biographies_with_sentiment_sample20k.csv
```
This runs sentiment analysis on a random 20,000-article sample.

Tehn run stats:

```bash
poetry run python scripts/run_stats.py --sample-n 20000
```
### Option B — Full Dataset (Slower)

```bash
poetry run python scripts/run_sentiment.py
poetry run python scripts/run_analysis.py
poetry run python scripts/run_stats.py
```
Outputs include:
- sentiment_summary_by_gender.csv
- pronoun_stats_by_gender.csv
- roberta_distribution_by_gender.csv
- stats_continuous_male_vs_female.csv
stats_chi2_roberta_male_vs_female.csv