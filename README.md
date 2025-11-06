# -Fake-News-Detection-using-Machine-Learning

# 🧠 Fake News Detection using Machine Learning

## 📌 Overview
This project uses machine learning to detect whether a news article is *real or fake* based on its title text.

## ⚙️ Tools Used
- Python
- scikit-learn
- TF-IDF Vectorization
- PassiveAggressiveClassifier

## 📊 Dataset
Dataset from [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

| Label | Meaning  |
|--------|-----------|
| 1 | Real News |
| 0 | Fake News |

## 🧠 Model
- Text data vectorized using TF-IDF
- Model: Passive-Aggressive Classifier
- Train/Test split: 80/20

*Accuracy:* ~94%

## 🗂 Files
- fake_news_detector.py — main script  
- sample_predictions.csv — output sample
