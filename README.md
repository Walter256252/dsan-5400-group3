# Gender Bias in Wikipedia Biographical Articles  
### DSAN 5400 – Group 3  
**Authors:** Laurent Julia Calac, Jiayuan Gong, Walter Hall, Jing Tan, Hung Tran  

---

# Project Overview

This project investigates **gender bias in Wikipedia biographical articles**, a widely studied issue in computational social science and NLP fairness. We examine whether the **sentiment**, **descriptive framing**, and **article structure** differ systematically between men’s and women’s biographies.

Using modern NLP models—including **transformer-based sentiment analysis**, we analyze more than **573,000** cleaned biographies.

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

## Option A: Download manually  
Download any files you'd like directly from the Drive folder.

## Option B: Use `--drive-url` to auto-download  
Example:


```bash
poetry run python scripts/run_sentiment.py \
  --drive-url "https://drive.google.com/file/d/FILE_ID/view?usp=sharing"

```
Replace FILE_ID with the exact file ID for the dataset you want to fetch.

---

# Data Source

Our project sources the previously gathered URLs of Wikipedia biographies from this [project](https://github.com/DavidGrangier/wikipedia-biography-dataset?tab=readme-ov-file) on neural text generation.
* insert citation

If you would like to scrape the data yourself using the URLs, please run:

```bash
python scrape_data.py \
    --input ../biography_urls.txt \
    --workers 5 \
    --outdir ../output
```

---

# Methods

## **1. Data Acquisition**
Scraped Wikipedia biography pages using `?curid=###` links from `biography_urls.txt`.

Tools used:
- `scrape_data.py`
- Multiprocessing workers
- Automatic rate limiting
- HTML parsing with BeautifulSoup

---

## **2. Cleaning & Normalization**
`clean_data.py` performs:
- Unicode normalization  
- Removal of references, templates, HTML tags, categories  
- Lowercasing and whitespace collapsing  
- Removal of maintenance templates  
- Length statistics (characters, words)

Output: data/processed/biographies_clean.csv

---

## **3. Gender Assignment**
`add_pronoun_gender.py` counts:
- Male pronouns: **he / him / his**
- Female pronouns: **she / her / hers**

Articles with strong pronoun usage are assigned gender labels.

---

## **4. Sentiment Analysis**

Three sentiment models:

| Model | Type | Output |
|-------|-------|---------|
| **VADER** | Lexicon-based | polarity scores |
| **TextBlob** | Rule-based | polarity + subjectivity |
| **RoBERTa (CardiffNLP)** | Transformer | negative / neutral / positive |

Script:run_sentiment.py


---

## **5. Exploratory Analysis**
`run_analysis.py` computes:
- Sentiment averages by gender  
- Pronoun statistics  
- Article length distributions  
- Transformer label frequencies  

---

## **6. Statistical Testing**
Using `run_stats.py`, we compute:
- Student’s t-test  
- Mann–Whitney U test  
- Kolmogorov–Smirnov test  
- Chi-square test for RoBERTa labels  

Results stored in: results/sentiment/


---

# Reproducibility Instructions

## **1. Install Dependencies**
```bash
poetry install
```
## **2. Ensure Cleaned Data Exists**

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

---

#### Step 1: Run Sentiment on 20k Sample
```bash
poetry run python scripts/run_sentiment.py \
  --sample-n 20000 \
  --output-path data/processed/biographies_with_sentiment_sample20k.csv
```
This runs sentiment analysis on a random 20,000-article sample.
This will output:
- Load biographies_clean.csv
- Randomly sample 20k rows
- Save results to: data/processed/biographies_with_sentiment_sample20k.csv

---

#### Step 2: Run Statistical Analysis on 20k Sample
Then run analysis:

```bash
poetry run python scripts/run_analysis.py \
    --input-path data/processed/biographies_with_sentiment_sample20k.csv
```

This will generate:
- sentiment_summary_by_gender.csv
- pronoun_stats_by_gender.csv
- roberta_distribution_by_gender.csv
All saved to: results/sentiment/

#### Step 3: Run Statistical Tests on 20k Sample
Finally, run stats:

```bash
poetry run python scripts/run_stats.py \
    --input-path data/processed/biographies_with_sentiment_sample20k.csv
```
Outputs include:
- stats_continuous_male_vs_female.csv
- stats_chi2_roberta_male_vs_female.csv



### Option B — Full Dataset (Complete Analysis, Slower)

Option B runs the entire pipeline on the **full set of 573,736 biographies**.  
This produces the final, publication-ready results.

Because of dataset size, RoBERTa sentiment analysis may take **many hours** on CPU. It took ~30 hours on a MAC laptop.

---

#### Step 1 — Run Full Sentiment Analysis
```bash
poetry run python scripts/run_sentiment.py
```
Outputs include: data/processed/biographies_with_sentiment.csv

---

#### Step 2 — Run Full Exploratory Analysis
```bash
poetry run python scripts/run_analysis.py
```
Outputs include:
- sentiment_summary_by_gender.csv
- pronoun_stats_by_gender.csv
- roberta_distribution_by_gender.csv
Saved to: results/sentiment/

---
#### Step 3 — Run Full Statistical Tests
```bash
poetry run python scripts/run_stats.py
```
Outputs include:
- stats_continuous_male_vs_female.csv
- stats_chi2_roberta_male_vs_female.csv
Saved to: results/sentiment/

## Summary of Outputs

| File | Description | Location |
|------|-------------|----------|
| **biographies_with_sentiment_sample20k.csv** | 20k sample with full sentiment analysis | `data/processed/` |
| **biographies_with_sentiment.csv** | Full dataset with sentiment analysis | `data/processed/` |
| **sentiment_summary_by_gender.csv** | Mean sentiment values by gender | `results/sentiment/` |
| **pronoun_stats_by_gender.csv** | Pronoun usage validation | `results/sentiment/` |
| **roberta_distribution_by_gender.csv** | Negative / neutral / positive label counts | `results/sentiment/` |
| **stats_continuous_male_vs_female.csv** | t-test, MWU, KS comparisons | `results/sentiment/` |
| **stats_chi2_roberta_male_vs_female.csv** | Chi-square tests for RoBERTa sentiment labels | `results/sentiment/` |


# Repository Structure

```
dsan-5400-group3/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── biographies_clean.csv
│       ├── biographies_with_sentiment.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── docs/
│
├── notebooks/
│   ├── cleaning.ipynb
│   ├── sentiment_analysis.ipynb
│   ├── statistical_testing.ipynb
│   └── exploratory_analysis.ipynb
│
├── results/
│   └── sentiment/
│       ├── sentiment_summary_by_gender.csv
│       ├── pronoun_stats_by_gender.csv
│       ├── roberta_distribution_by_gender.csv
│       ├── stats_continuous_male_vs_female.csv
│       └── stats_chi2_roberta_male_vs_female.csv
│
├── scripts/
│   ├── scrape_data.py
│   ├── clean_data.py
│   ├── add_pronoun_gender.py
│   ├── run_sentiment.py
│   ├── run_analysis.py
│   └── run_stats.py
│
└── src/dsan_5400_group3/
    ├── preprocessing.py
    ├── sentiment.py
    ├── evaluation.py
    └── utils.py
```

---

# Summary of Findings

Using **573,736** cleaned Wikipedia biographies, we identify several robust patterns of gender differences in sentiment, subjectivity, and linguistic structure.

---

## 1. Female biographies contain more positive sentiment

| Model              | Male Avg | Female Avg | Interpretation                     |
|-------------------|----------|------------|-------------------------------------|
| **VADER compound** | 0.6395   | 0.7650     | Higher positivity for women         |
| **TextBlob polarity** | 0.0776   | 0.1026     | More positive language overall      |
| **RoBERTa labels** | χ² ≈ **1218** (p < 10⁻²⁶⁰) | — | Strong shift in sentiment distribution |

All three models agree:  
**Female biographies are written with more positive emotional tone.**

---

## 2. Female biographies are more subjective

**TextBlob subjectivity scores:**

- **Male:** 0.322  
- **Female:** 0.338  

Female biographies use slightly more opinionated or descriptive language.

---

## 3. Pronoun statistics strongly validate gender assignment

| Metric               | Male Bios | Female Bios |
|----------------------|-----------|-------------|
| **Male pronouns**    | 22.4      | 2.2         |
| **Female pronouns**  | 0.55      | 24.6        |

This confirms that our pronoun-based gender detection is highly reliable.

---

## 4. Female biographies are slightly longer

- **Male biographies:** ~1478 words  
- **Female biographies:** ~1549 words  

This pattern aligns with prior research showing more editorial attention to women in certain domains.

---

## 5. Results remain stable in the 20k sample

The 20,000-article sample reproduces the same direction and magnitude of effects:

- Sampling is reliable  
- Differences are robust  
- Results are not artifacts of sample size  

---

# Gender Imbalance & Why It Matters

Wikipedia’s gender distribution is highly uneven:

- **~77% male biographies**
- **~15% female biographies**
- **Remainder: unknown or nonbinary**

This imbalance:

- **is the phenomenon being studied**, not a methodological flaw  
- **reflects real structural biases** in coverage and notability  
- emphasizes the importance of computational analyses of representation  

The imbalance is consistent across:

- full dataset  
- 20k sample subset  
- historical Wikipedia snapshots  

---

## Citation

If referencing this work:

**Calac, L. J., Gong, J., Hall, W., Tan, J., & Tran, H. (2025).  
*Gender Bias in Wikipedia Biographical Articles.*  
DSAN 5400, Georgetown University.**
