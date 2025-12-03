# Discussion

The goal of this project was to examine whether Wikipedia biographies differ systematically in sentiment and linguistic framing based on the gender of the person being described. Using a large corpus of **573,736 cleaned biographies** and three sentiment analysis systems (VADER, TextBlob, and RoBERTa), we find consistent and statistically significant gender differences in sentiment, subjectivity, and pronoun usage. These differences are robust across all statistical tests, including Welch t-tests, Mann–Whitney U tests, Kolmogorov–Smirnov distributional tests, and chi-square tests for categorical sentiment labels.

## Gender Imbalance in Wikipedia Biographies

A key characteristic of the dataset is its **strong gender imbalance**:

- **442,288 male biographies (77.1%)**  
- **85,305 female biographies (14.9%)**  
- **Remaining biographies labeled “unknown” or missing gender**

This imbalance is **not a flaw** of the dataset—it reflects a well-documented structural feature of Wikipedia. Prior research shows that approximately **80–85% of biographies on Wikipedia are about men**, due to:

- historical inequalities in who becomes notable;  
- systemic bias in editorial coverage;  
- demographic skew among Wikipedia editors (majority male);  
- greater documentation and media attention given to male figures.

Because Wikipedia's demographic composition is inherently imbalanced, our dataset **faithfully represents the real-world information ecosystem** we aim to evaluate.

### Why the imbalance does *not* distort our findings

1. **Sample size is large enough across genders:**  
   Even though female biographies are fewer, *85,305* female entries provide more than enough statistical power to detect meaningful differences.

2. **Bias itself is part of the phenomenon:**  
   Gender imbalance is *informative*, not an artifact. It reflects the cultural and editorial context in which Wikipedia biographies are written.

3. **Sentiment differences remain consistent in smaller, balanced subsamples:**  
   Preliminary tests using random 20k samples showed the *same* sentiment-direction differences as the full dataset.

4. **The analysis focuses on per-biography sentiment, not aggregate volume:**  
   Our statistical comparisons rely on:
   - means  
   - medians  
   - distributions  
   - proportions  
   …which remain valid regardless of group size differences.

**Conclusion:**  
The gender imbalance accurately represents Wikipedia’s own structural bias, and the size of each subgroup is sufficient for rigorous statistical comparison. The observed sentiment differences are real, robust, and not artifacts of unequal group size.

---

## 1. Sentiment Differences by Gender

Across all three sentiment systems—VADER (lexicon-based), TextBlob (rule-based), and RoBERTa (transformer-based)—female biographies exhibit **more positive sentiment** than male biographies. The magnitude of these differences is statistically large given the enormous sample size.

### VADER Compound Sentiment
- Male mean = **0.6395**  
- Female mean = **0.7650**  
- *t*(>500k) = **–58.13**, *p* < 10⁻³⁰⁰  

### TextBlob Polarity
- Male mean = **0.0776**  
- Female mean = **0.1026**  
- *t* = **–90.79**, *p* < 10⁻³⁰⁰  

### Subjectivity
- Male mean = **0.3224**  
- Female mean = **0.3382**  
- *t* = **–46.60**, *p* < 10⁻³⁰⁰  

All results indicate an extremely small probability that these differences are due to chance. Mann–Whitney U and KS tests confirm broader distributional differences.

**Interpretation:**  
Female biographies are written using more positive and slightly more subjective language. This pattern aligns with findings in previous studies on gendered language in Wikipedia.

---

## 2. RoBERTa Sentiment Label Differences

A chi-square test of independence on RoBERTa sentiment labels (negative/neutral/positive) also reveals strong gender differences:

- χ² = **1218.73**, df = 2, *p* ≈ 2.3×10⁻²⁶⁵  

### Contingency Table
| Gender | Negative | Neutral | Positive |
|--------|----------|---------|----------|
| Female |   698    | 80,765  | 3,842    |
| Male   |  4,284   | 427,443 | 10,561   |

**Interpretation:**  
Female biographies receive proportionally fewer negative labels and more positive labels compared to male biographies. This reinforces the continuous sentiment findings.

---

## 3. Pronoun Usage Shows Expected but Extreme Asymmetries

Pronoun counts not only validate the gender tagging pipeline but also reveal strong structural patterns:

- **Male pronouns per male biography:** 22.45  
- **Male pronouns per female biography:** 2.21  

- **Female pronouns per female biography:** 24.60  
- **Female pronouns per male biography:** 0.55  

Both tests yield *t*-statistics exceeding **180** and effectively zero p-values.  
KS statistics between 0.68 and 0.86 indicate fundamentally different distributions.

**Interpretation:**  
Pronoun usage strongly reflects the subject’s gender and shows that the linguistic framing of biographies is deeply gendered. These counts also serve as important control variables.

---

## 4. Article Length Differences

Female biographies are **longer on average**:

- Male: **1,477.8 words**  
- Female: **1,549.3 words**  
- *t* = –8.97, *p* = 3×10⁻¹⁹  

**Interpretation:**  
This may reflect increased editorial attention to female biographies through initiatives such as “Women in Red,” which seek to redress Wikipedia’s gender gaps. Longer biographies may contribute to differences in sentiment richness.

---

## 5. Overall Interpretation

Across every metric—lexicon sentiment, rule-based polarity, transformer sentiment labels, subjectivity, article length, and pronoun usage—we find robust gender differences in the language used in Wikipedia biographies.

These patterns suggest:

1. **More positive linguistic framing** in biographies about women.  
2. **More narrative or subjective language** in female biographies.  
3. **Structurally gendered writing patterns**, including pronoun asymmetry.  
4. **Potential editorial compensation**, where editors consciously or unconsciously use more positive framing when writing about women.  
5. **Different topical distributions**, where women’s biographies may overrepresent areas with inherently more positive or narrative wording.

Importantly, the direction and strength of these results remain stable even when considering the natural gender imbalance in the dataset.

---

## 6. Recommendations for Future Work

Future analyses could extend this study by:

- Applying **topic modeling** to understand how occupation intersects with gendered sentiment.  
- Fitting **multivariate regression models** to control for article length, pronoun usage, and topic category.  
- Conducting **temporal analyses** to determine if sentiment differences have grown or shrunk over time.  
- Using **fine-tuned BERT or domain-specific models** to improve sentiment accuracy on encyclopedic text.  
- Exploring **local explanations** (e.g., SHAP values or keyword salience) to identify the specific kinds of language that differ by gender.
