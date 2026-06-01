# nlp-text-classification

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![NLTK](https://img.shields.io/badge/NLTK-3776AB?style=flat&logo=python&logoColor=white)

A series of NLP experiments covering the full pipeline — from raw text preprocessing through classical ML, deep learning (LSTM, RNN), and transformer-based models (GPT-2). Built as part of a university NLP course (Afeka College of Engineering).

---

## Projects

### Final Project — News Source Analysis (BBC vs CNN)

Comparative NLP analysis of BBC and CNN news corpora to identify linguistic patterns and key topic differences between the two sources.

**What it does:**
- Scrapes and preprocesses real news data from BBC and CNN
- Builds and compares TF-IDF word importance profiles per source
- Trains Word2Vec embeddings to model semantic relationships
- Uses an Autoencoder on Word2Vec embeddings to surface the most semantically significant terms
- Visualizes results with bar charts and word clouds

**Key techniques:** TF-IDF · Word2Vec · Autoencoder (Keras) · spaCy tokenization · lemmatization · WordCloud

---

### Task 1 — Spam Detection & Text Statistics

Exploratory analysis on the classic SMS Spam Collection dataset.

**What it does:**
- Counts spam vs. ham message distributions
- Computes word frequency statistics and unique word counts
- Compares NLTK vs. spaCy pipelines for tokenization, stopword removal, lemmatization, and stemming
- Side-by-side benchmark of both NLP libraries on the same corpus

**Key techniques:** NLTK · spaCy · tokenization · lemmatization · stemming · FreqDist

---

### Task 2 — Tokenization & Preprocessing Comparison

Systematic comparison of tokenization strategies on raw text data.

**What it does:**
- Implements and compares whitespace tokenizer, regex tokenizer, NLTK word tokenizer, and sentence tokenizer
- Applies stemming (Porter) and lemmatization (WordNet) side by side
- Time complexity analysis of each preprocessing approach

**Key techniques:** NLTK · spaCy · regex · PorterStemmer · WordNetLemmatizer

---

### Task 3 — Sequence Modeling & Text Generation

Deep learning models for text classification and generation on preprocessed text.

**What it does:**
- Full text normalization pipeline (lowercasing, contraction expansion, stopword removal, lemmatization)
- TF-IDF vectorization of sentences
- Trains LSTM and SimpleRNN models for sequence classification
- Fine-tunes GPT-2 (HuggingFace Transformers) for text generation

**Key techniques:** TF-IDF · LSTM · SimpleRNN · GPT-2 · HuggingFace Transformers · TensorFlow/Keras · pad_sequences

---

## Tech Stack

| Category | Libraries |
|---|---|
| NLP & preprocessing | spaCy, NLTK, contractions |
| Classical ML | scikit-learn (TF-IDF, vectorizers) |
| Word embeddings | Gensim (Word2Vec) |
| Deep learning | TensorFlow / Keras (LSTM, RNN, Autoencoder) |
| Transformers | HuggingFace Transformers (GPT-2) |
| Data & visualization | pandas, numpy, matplotlib, WordCloud |

---

## Project Structure

```
nlp-text-classification/
├── Final project Twitwe/
│   ├── final_nlp.ipynb          # BBC vs CNN analysis — TF-IDF, Word2Vec, Autoencoder
│   ├── bbc.csv                  # BBC news dataset
│   └── cnn.csv                  # CNN news dataset
├── task_1/
│   ├── Untitled.ipynb           # Spam detection & NLTK vs spaCy comparison
│   └── spam.csv                 # SMS Spam Collection dataset
├── task_2/
│   ├── nlp_h2.ipynb             # Tokenization strategies comparison
│   └── data.txt                 # Raw text corpus
└── task_3/
    ├── h3.ipynb                 # LSTM, RNN, GPT-2 text generation
    └── data.txt                 # Raw text corpus
```

---

## Getting Started

### Prerequisites

```bash
pip install spacy nltk gensim tensorflow scikit-learn transformers torch
pip install pandas numpy matplotlib wordcloud ntscraper contractions
python -m spacy download en_core_web_sm
```

### Run

Open any notebook in Jupyter and run cells in order. Each notebook is self-contained.

```bash
jupyter notebook
```

---

## Author

**Tal Melnikov** — [LinkedIn](https://www.linkedin.com/in/tal-melnikov/) · [GitHub](https://github.com/talmelnikov)
