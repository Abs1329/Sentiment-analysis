import pandas as pd
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

data = {
    'review': [
        "I love this phone, it has great features!",
        "Worst laptop ever. Waste of money.",
        "The camera quality is amazing and battery life is good.",
        "This product is terrible, I regret buying it.",
        "Excellent headphones, crystal clear sound!",
        "Bad experience. The item broke in two days.",
        "Not bad, but could be better.",
        "Absolutely fantastic! Highly recommend.",
        "Poor build quality, not worth the price.",
        "I am very happy with this purchase."
    ],
    'sentiment': [
        'positive', 'negative', 'positive', 'negative', 'positive',
        'negative', 'neutral', 'positive', 'negative', 'positive'
    ]
}

df = pd.DataFrame(data)
print(df.head())
print()

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

df['cleaned_review'] = df['review'].apply(preprocess_text)
print(df[['review', 'cleaned_review']].head())
print()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_review'])
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(accuracy_score(y_test, y_pred))
print()
print(classification_report(y_test, y_pred))
print()
print(confusion_matrix(y_test, y_pred))
print()

def predict_sentiment(review_text):
    cleaned = preprocess_text(review_text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    return prediction

test_reviews = [
    "This phone is awesome, I use it every day!",
    "Very disappointed. Waste of money.",
    "It works okay, nothing special."
]

for r in test_reviews:
    print(f"Review: {r} --> Sentiment: {predict_sentiment(r)}")

user_input = input("Enter a product review: ")
print("Predicted Sentiment:", predict_sentiment(user_input))
