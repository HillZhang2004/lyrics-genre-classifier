# Lyrics Genre Classifier

Predicts a song’s genre from **lyrics text only** (no audio).  
Labels: **Folk, Jazz, Metal, Pop, Rock**.

**TL;DR takeaway:** A strong classic baseline (TF-IDF) matches or beats “modern” embeddings on this task, and most errors collapse into **Rock** (the model’s “magnet” class).

## Quick links (for recruiters)

Slides (project overview + results):
https://docs.google.com/presentation/d/10pTTQb_L21_sDHcCswdqTa61122DklkoD4eUYz3HnkM/edit?usp=sharing

Key visuals (in this repo):
- PCA of MiniLM embeddings: `assets/figures/pca_minilm_embeddings.png`
- Normalized confusion matrix (TF-IDF + LR): `assets/figures/confusion_matrix_tfidf_lr.png`
- Top TF-IDF words (example: Folk): `assets/figures/top_tfidf_words_folk.png`

## Results (held-out test set)

I keep the classifier fixed (**multiclass logistic regression**) and only change the feature representation.

| Features | Test accuracy | Test macro-F1 |
|---|---:|---:|
| TF-IDF (20k vocab) | 0.6096 | 0.4732 |
| MiniLM embeddings (384d) | 0.6090 | 0.4426 |
| MiniLM + Autoencoder (64d latent) | 0.5939 | 0.3938 |

**Interpretation:** Accuracy is similar for TF-IDF and MiniLM, but TF-IDF wins on macro-F1. The autoencoder compression loses signal for this label set.

## Data and split

Dataset: Kaggle “Multi-Lingual Lyrics for Genre Classification” (Matei Bejan)  
https://www.kaggle.com/datasets/mateibejan/multilingual-lyrics-for-genre-classification

In my runs, I use `train.csv` only and rename it to:
- `data/lyrics_train.csv`

Split (stratified):
- Train: 30k
- Validation: 10k (tune regularization C using macro-F1)
- Test: 10k (final report)

## What I compared

All models use logistic regression; only the representation changes:
1. TF-IDF (bag of words, capped at ~20k features)
2. MiniLM sentence embeddings (384-d)
3. MiniLM embeddings compressed with an autoencoder (64-d latent)

## Repo structure

- `notebooks/lyrics_genre_project.ipynb`  
  End-to-end pipeline: preprocessing → features → training → evaluation → plots.

- `assets/figures/`  
  Saved figures used in the README / slides.

- `data/`  
  Local-only (not tracked). Put `lyrics_train.csv` here.

## How to run

1. Download the dataset from Kaggle.
2. Put `train.csv` into `data/` and rename to `lyrics_train.csv`.
3. Run the notebook:
   - `notebooks/lyrics_genre_project.ipynb`

Typical dependencies:
- pandas, numpy, scikit-learn, matplotlib
- sentence-transformers, torch

## Notes / what I learned

- TF-IDF is a surprisingly strong baseline for lyric-only classification.
- The confusion matrix shows Rock absorbs many mistakes, suggesting lyric style features overlap heavily across genres.
- Embeddings are not automatically “better”; representation choice depends on the task and evaluation metric (macro-F1 mattered here).
