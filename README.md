# Hybrid Mental Health Detection Using DistilBERT + DASS-21 + SVM

## Overview
This project proposes a hybrid model for multi-class mental health detection from text data, combining DistilBERT embeddings with DASS-21 psychological scores and SVM classification.

## Model Architecture
- **DistilBERT** → 768-dim CLS embeddings
- **DASS-21** → 2-dim proxy scores (depression + anxiety)
- **Hybrid Feature** → 770-dim (768 + 2)
- **Classifier** → LinearSVC with 5-Fold Cross Validation

## Dataset
- Source: [Sentiment Analysis for Mental Health](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
- Classes: Normal, Depression, Anxiety, Suicidal
- Total samples: ~46,240

## Results
| Metric | Score |
|--------|-------|
| Test Accuracy | 79.41% |
| Macro F1 | 0.79 |
| CV Mean (5-Fold) | 0.7951 ± 0.0062 |

## Class-wise Performance
| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Anxiety | 0.88 | 0.80 | 0.84 |
| Depression | 0.72 | 0.75 | 0.73 |
| Normal | 0.91 | 0.96 | 0.94 |
| Suicidal | 0.67 | 0.61 | 0.63 |

## Requirements
transformers
torch
scikit-learn
pandas
numpy
joblib
matplotlib
