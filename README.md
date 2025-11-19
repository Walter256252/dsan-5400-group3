# Gender Bias in Wikipedia Biographical Articles  
### DSAN 5400 – Group 3  
**Authors:** Laurent Julia Calac, Jiayuan Gong, Walter Hall, Jing Tan, Hung Tran

---

## Project Overview

This project investigates **gender bias in Wikipedia biographical articles**. Prior studies show measurable disparities between how men and women are portrayed in open knowledge platforms. Our goal is to evaluate whether **sentiment**, **topic distribution**, and **linguistic framing** differ systematically between biographies of men and women on Wikipedia.

Understanding these patterns provides insights into representation in public knowledge sources and supports more equitable uses of Wikipedia-derived text in downstream AI systems.

---

##  Research Question

> **Do Wikipedia biographies of men and women differ in sentiment, topic distribution, or linguistic framing?**

We measure differences in:
- Sentiment polarity  
- Topical focus  
- Word category usage (e.g., personal life vs. career)  
- Syntactic and lexical framing  

---

## Related Work

Previous research has shown consistent asymmetries in Wikipedia coverage:

- **Wagner et al. (2015)** – Women’s biographies emphasize family/relationships more often than men’s.
- **Graells-Garrido et al. (2015)** – Topic distribution and network centrality differ by gender.
- **Field et al. (2021)** – Even after controlling for profession/notability, subtle sentiment and framing differences persist.

Our project builds upon these studies using updated NLP methods, particularly **transformers** and modern topic modeling techniques.

---

## Data Sources

We use the **Wikipedia Biography Dataset**, or scrape our own dataset using Wikimedia dumps and Wikidata labels.

Our final dataset includes:

- Full text of biographical pages  
- Metadata:
  - Gender  
  - Occupation  
  - Birth year  
  - Article length  
  - Page view statistics  
  - Revision history  
- Derived features:
  - LIWC-style semantic categories  
  - Sentiment scores  
  - Topic model vectors  

Raw data is too large for GitHub and must be downloaded locally using provided scripts.

---

##  Methods

### **1. Named Entity Recognition**
- spaCy NER to extract entities related to occupation, relationships, organizations, etc.

### **2. Sentiment Analysis**
- **Lexicon-based**: VADER/TextBlob  
- **Transformer-based**: CardiffNLP RoBERTa sentiment model  

### **3. Word Category Analysis (LIWC-style)**
- Custom lexicon mapping  
- Frequency analysis for categories such as:
  - Family  
  - Appearance  
  - Career  
  - Emotion  

### **4. Topic Modeling**
- LDA (gensim)  
- BERTopic (sentence-transformer embeddings + clustering)  

### **5. Syntactic & Lexical Analysis**
- POS tagging  
- Dependency parsing  
- Adjective and pronoun frequency  
- Subject descriptors and salient terms  

---

## Evaluation

### **Quantitative**
- t-tests, chi-square tests, Mann–Whitney U  
- Effect sizes  
- Normalization for article length and occupation  

### **Qualitative**
- Examine representative sentences  
- Compare adjectives used to describe men vs. women  
- Manual validation of model-identified patterns  

### Baselines
- Gender-agnostic averages for sentiment, word categories, and topics  

---

## Repository Structure

```text
dsan-5400-group3/
│
├── data/                  # raw and processed data
├── docs/                 # documentation and Sphinx site
├── notebooks/             # exploratory and analysis notebooks
    ├── scraping.ipynb
    ├── cleaning.ipynb
    ├── sentiment_analysis.ipynb
    ├── topic_modeling.ipynb
    └── bias_metrics.ipynb        
├── results/               # figures, tables, trained models
├── scripts/               # CLI scripts for reproducibility
    ├── download_data.py
    ├── clean_data.py
    ├── run_sentiment.py
    ├── run_topic_modeling.py
    ├── run_analysis.py             
├── src/dsan_5400_group3/  # main Python package
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── sentiment.py
│   ├── topic_modeling.py
│   ├── framing_analysis.py
│   ├── evaluation.py
│   └── utils.py
└── tests/ 
    └── test_dsan_5400_group3.py               # unit tests


