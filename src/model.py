# =========================================
# SENTIMENT ANALYSIS - MODEL PIPELINE
# =========================================

import pandas as pd
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

# Download stopwords
nltk.download('stopwords')

# =========================================
# 1. LOAD DATA
# =========================================
def load_data(path):
    df = pd.read_csv(path)
    df = df[['Text', 'Score']].dropna()
    return df

# =========================================
# 2. LABEL SENTIMENT
# =========================================
def label_sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

# =========================================
# 3. PREPROCESS TEXT
# =========================================
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = "".join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# =========================================
# 4. FEATURE EXTRACTION
# =========================================
def vectorize_text(df):
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['CleanText'])
    return X, vectorizer

# =========================================
# 5. TRAIN MODELS
# =========================================
def train_models(X_train, y_train):
    nb = MultinomialNB()
    lr = LogisticRegression(max_iter=200)

    nb.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    return nb, lr

# =========================================
# 6. EVALUATION
# =========================================
def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    print(f"\n===== {model_name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{model_name} - Confusion Matrix")
    plt.show()

    return y_pred

# =========================================
# 7. INSIGHTS FUNCTIONS
# =========================================
def sentiment_distribution(df):
    print("\nSentiment Counts:\n", df['Sentiment'].value_counts())

    sns.countplot(x='Sentiment', data=df)
    plt.title("Sentiment Distribution")
    plt.show()

def review_length_analysis(df):
    df['ReviewLength'] = df['CleanText'].apply(len)

    sns.histplot(df['ReviewLength'], bins=50)
    plt.title("Review Length Distribution")
    plt.show()

    print("\nAverage Review Length:")
    print(df.groupby('Sentiment')['ReviewLength'].mean())

def common_words(df):
    words = " ".join(df['CleanText']).split()
    common = Counter(words).most_common(20)

    words, counts = zip(*common)

    plt.figure(figsize=(10,5))
    plt.bar(words, counts)
    plt.xticks(rotation=45)
    plt.title("Top 20 Words")
    plt.show()

# =========================================
# 8. MAIN PIPELINE
# =========================================
def main():
    # Load data
    df = load_data("../data/Reviews.csv")

    # Label sentiment
    df['Sentiment'] = df['Score'].apply(label_sentiment)

    # Insights before processing
    sentiment_distribution(df)

    # Preprocess text
    df['CleanText'] = df['Text'].apply(preprocess)

    # More insights
    review_length_analysis(df)
    common_words(df)

    # Vectorize
    X, vectorizer = vectorize_text(df)
    y = df['Sentiment']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    nb_model, lr_model = train_models(X_train, y_train)

    # Evaluate
    evaluate(nb_model, X_test, y_test, "Naive Bayes")
    evaluate(lr_model, X_test, y_test, "Logistic Regression")

# =========================================
# RUN SCRIPT
# =========================================
if __name__ == "__main__":
    main()
