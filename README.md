# Lyrics Genre Classifier

This project predicts a song’s genre using **lyrics text only** (no audio).  
Given lyrics, the model predicts one of five genres: **Folk, Jazz, Metal, Pop, Rock**.

I use this setup to test how much genre signal exists in lyrics alone, and to compare a strong TF-IDF baseline against “modern” text features (MiniLM embeddings, with and without an autoencoder).

## Dataset

I use the Kaggle dataset **“Multi-Lingual Lyrics for Genre Classification”** by Matei Bejan:  
https://www.kaggle.com/datasets/mateibejan/multilingual-lyrics-for-genre-classification

Important: after downloading, I only use the **train.csv** file (the large file).  
In my local setup, I rename it to:

- `lyrics_train.csv`

and place it here:

- `data/lyrics_train.csv`

The raw dataset is not included in this GitHub repo because it is large (hundreds of MB) and GitHub does not support pushing files that big by default.

## Repo structure

- `notebooks/lyrics_genre_project.ipynb`  
  Main notebook with preprocessing, feature extraction, training, and evaluation.

- `data/`  
  Not tracked in git. This is where you should put `lyrics_train.csv`.

## How to run

1. Download the dataset from Kaggle (link above).
2. Put the file in `data/lyrics_train.csv`.
3. Open and run:
   - `notebooks/lyrics_genre_project.ipynb`

Typical dependencies include: pandas, numpy, scikit-learn, matplotlib, sentence-transformers, torch.

## What I compared

All comparisons use logistic regression as the classifier, and only the feature representation changes:

1. TF-IDF features (bag of words)
2. MiniLM embeddings (384 dim)
3. MiniLM embeddings compressed with an autoencoder (64 dim latent)

I tune regularization on the validation set using macro-F1, then report final test accuracy and macro-F1 plus confusion matrices.

## Notes

Results depend on the exact random seed and split, but the main takeaway was consistent in my final run: TF-IDF is a very strong baseline for this task.
