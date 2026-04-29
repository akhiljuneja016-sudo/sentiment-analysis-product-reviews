# 📊 Sentiment Analysis on Product Reviews

## 📌 Project Overview

This project performs **Sentiment Analysis** on product reviews by classifying them into:

* Positive 😊
* Negative 😡
* Neutral 😐

The goal is to extract meaningful insights from customer feedback using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

---

## 🎯 Problem Statement

To develop a machine learning model that analyzes product reviews and accurately predicts their sentiment, enabling businesses to understand customer opinions and improve decision-making.

---

## 💡 Motivation

Online platforms like Amazon and Flipkart generate massive amounts of customer reviews daily. Manually analyzing them is inefficient.

This project helps:

* Understand customer satisfaction
* Improve product quality
* Gain market insights
* Automate feedback analysis

---

## 🧠 Methodology

### 🔹 Workflow

1. Data Collection (Reviews.csv dataset)
2. Data Preprocessing (cleaning text, removing stopwords)
3. Feature Extraction (TF-IDF Vectorization)
4. Model Training
5. Evaluation and Analysis
6. Visualization

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Matplotlib
* Seaborn

---

## 📂 Dataset

* File: `Reviews.csv`
* Contains product reviews and ratings
* Key columns:

  * `Text` → Review content
  * `Score` → Rating (1–5)

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/sentiment-analysis.git
cd sentiment-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the script or notebook:

```bash
python src/model.py
```

or open:

```
notebook/sentiment_analysis.ipynb
```

---

## 📊 Results & Insights

### ✔️ Key Outputs:

* Sentiment distribution analysis
* Review length insights
* Most frequent words
* Model performance comparison
* Confusion matrix visualization

### ✔️ Models Used:

* Naive Bayes
* Logistic Regression

### ✔️ Performance:

* Achieved high accuracy (~80–90%)
* Logistic Regression slightly outperformed Naive Bayes

---

## 📈 Visualizations

* Bar charts (Sentiment distribution)
* Histogram (Review length)
* Heatmap (Confusion matrix)
* Word frequency plots

---

## 🔍 Sample Insights

* Majority of reviews are positive
* Negative reviews contain strong sentiment words
* Longer reviews tend to express clearer opinions

---

## 🚀 Future Scope

* Implement Deep Learning models (LSTM, BERT)
* Real-time sentiment analysis dashboard
* Multi-language support
* Aspect-based sentiment analysis

---

## 🤝 Contribution

This project is developed as part of an academic NLP assignment. Contributions and suggestions are welcome.

---

## 📜 License

This project is for educational purposes only.

---
