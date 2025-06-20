# 📩 Email Spam Classifier

A machine learning project to classify SMS messages as **Spam** or **Ham** using Natural Language Processing (NLP) techniques and various classification models.

---

## 🚀 Project Overview

This project uses a labeled SMS dataset to:
- Clean and preprocess text messages
- Extract relevant features
- Train multiple classification models
- Compare their performance
- Save the best model and vectorizer for deployment

---

## 📂 Dataset

The dataset used is `spam.csv`, which contains over 5,500 SMS messages labeled as `ham` (non-spam) or `spam`.

---

## ✅ Features Extracted

- `num_characters`: Total number of characters in message
- `num_words`: Total number of words
- `num_sentences`: Total number of sentences
- `transform_text`: Cleaned and stemmed version of message text
- TF-IDF vectors: Extracted using `TfidfVectorizer` with `max_features=3000`

---

## 🔧 Tech Stack

- **Language:** Python
- **Libraries:** 
  - `pandas`, `numpy` for data handling
  - `nltk`, `string` for NLP
  - `matplotlib`, `seaborn` for visualization
  - `sklearn`, `xgboost` for ML models
  - `wordcloud` for visualizing top words
- **Models Used:**
  - Naive Bayes (Multinomial, Bernoulli, Gaussian)
  - SVM, Logistic Regression, Random Forest, XGBoost, etc.
  - Ensemble: Voting Classifier and Stacking Classifier

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision** (most important due to spam detection)
- **Confusion Matrix**

> 📌 ExtraTreesClassifier achieved highest performance:
> - Accuracy: **97.87%**
> - Precision: **97.5%**

---

## 🧠 NLP Preprocessing

Custom text cleaning using:
- Lowercasing
- Tokenization
- Removing punctuation and stopwords
- Stemming with `PorterStemmer`
- Feature extraction via TF-IDF

---

## 📦 Saved Models

- `vectorizer.pkl`: Trained TF-IDF vectorizer
- `model.pkl`: Trained ML model (e.g., MultinomialNB or ExtraTrees)

These files can be loaded in a deployment app using Flask or Streamlit.

---

## 📈 Visualizations

- Word clouds for spam and ham messages
- Histogram plots of message lengths and word counts
- Top 30 most common words (spam & ham)
- Model comparison bar chart (Accuracy vs. Precision)

---

## 🔮 Future Improvements

- Use deep learning models like LSTM or BERT
- Handle data imbalance with SMOTE or resampling
- Deploy as web app (Flask + Streamlit)
- Add user interface for message input and prediction
