# NLP & Text Mining Labs

> Hands-on natural language processing and text mining labs completed as part of CS4083 at Effat University.

---

## Labs Overview

### Lab 1 — Data Analysis with Pandas: Goodreads Dataset
**File:** `NLP_Lab1.ipynb`

Cleaned and analyzed a dataset of ~6,000 "best books" scraped from Goodreads, practicing the core EDA workflow used throughout the NLP course.

**Dataset columns:** `rating`, `review_count`, `isbn`, `booktype`, `author_url`, `year`, `genre_urls`, `rating_count`, `name`

**Key techniques:**
- Loading CSV files without headers and manually assigning column names
- Cleaning malformed rows and fixing data types
- Parsing composite columns into atomic properties
- Grouping by genre and author, aggregating rating statistics
- Visualizing distributions with Matplotlib and Seaborn

**Tools:** Python · Pandas · NumPy · Matplotlib · Seaborn

---

### Lab 3 — Regex & Word Embeddings: Customer Reviews Analysis
**File:** `NLP_Lab3.ipynb`  
**Course:** CS4083 · Instructor: Dr. Naila Marir

Applied regex-based information extraction and word embedding techniques to analyze real e-commerce customer reviews from an online electronics store.

**Part 1 — Regex Information Extraction (40 pts):**
- Extracted product names, quantities, and prices from unstructured review text
- Converted word-numbers to digits (e.g. "three" → 3)
- Generated summary reports: total products sold, total revenue, most expensive product, highest-revenue product
- Validated email addresses and multi-format phone numbers (`xxx-xxx-xxxx`, `(xxx) xxx-xxxx`, `xxx.xxx.xxxx`)

**Part 2 — Word Embeddings Analysis (60 pts):**
- Preprocessed reviews: lowercasing, punctuation removal, stopword filtering, tokenization using NLTK
- Applied word embedding techniques for semantic similarity analysis
- Explored vector representations of product-related vocabulary

**Tools:** Python · Regex · NLTK · Scikit-learn · Pandas

---

### Lab 4 — Data-Centric vs Model-Centric AI: Review Classification
**File:** `NLP_Lab4.py`

Explored whether improving data quality or improving model architecture produces greater accuracy gains for a text classification task on magazine product reviews.

**Key techniques:**
- Baseline SGD classifier pipeline: `CountVectorizer → TfidfTransformer → SGDClassifier`
- Model-centric improvement: MultinomialNB with unigram + bigram TF-IDF (`ngram_range=(1,2)`)
- Data-centric improvement: regex-based detection and removal of HTML-noisy training rows (`<tags>`, `&lt;`, `href=`, etc.)
- Side-by-side accuracy comparison across both models before and after cleaning

**Key finding:** Cleaning corrupted training data produced larger accuracy improvements than switching model architecture — demonstrating the core principle of data-centric AI.

**Tools:** Python · Scikit-learn · Pandas · Regex

---

### Hackathon — Research Papers Dataset
**File:** `hackathon.ipynb`

End-to-end ML pipeline built on an academic research papers CSV (`papers.csv`) during a timed NLP hackathon.

**Steps covered:**
- Data loading with robust CSV parsing (`engine='python'`, `on_bad_lines='skip'`)
- Inspection: shape, dtypes, describe, missing values
- Feature engineering: dropped irrelevant columns (`id`, `event_type`, `pdf_name`, `paper_text`), geospatial distance calculation using the Haversine formula
- EDA with Matplotlib and Seaborn: distributions, correlations
- Linear Regression modeling with StandardScaler, train/test split, MSE and R² evaluation

**Tools:** Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn · SciPy

---

## Skills Demonstrated

| Area | Skills |
|------|--------|
| Text Processing | Regex, TF-IDF, Tokenization, Stopword removal, Word Embeddings |
| Data Wrangling | Pandas EDA, missing value handling, column parsing, noisy data detection |
| Classification | SGD, Naive Bayes, pipeline construction, accuracy evaluation |
| Information Extraction | Product/price/quantity extraction, email & phone validation |
| Feature Engineering | Geospatial features (Haversine), n-gram features |
| Data-Centric AI | HTML noise detection, data cleaning vs model tuning comparison |

---

## Course
**CS4083** — Text Mining & Natural Language Processing  
Instructor: Dr. Naila Marir  
Effat University · College of Engineering · Department of Computer Science

## Author
**Jouri Aldaghma**  
Computer Science (AI Concentration) · Effat University  
[LinkedIn](https://linkedin.com/in/jouri-aldaghma-b7a8b8307) · [GitHub](https://github.com/Jourialdagh)
