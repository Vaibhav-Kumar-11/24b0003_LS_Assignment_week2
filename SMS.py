import pandas as pd

df = pd.read_csv(r'D:\SUMMERS SORTED\LEARNERS SPACE\ML\Week 2\Q2_SMS\spam.csv', encoding='latin-1',names=["label", "message"])

import nltk
from nltk.corpus import stopwords

stop_words_set = set(stopwords.words('english'))

def preprocessing(text):
    text = text.lower()
    tokenized_words = text.split() # better approach nltk but kuch to issues arrhe the so switched at the end moment (will look into it personally in week 3 too.)

    output = []
    for word in tokenized_words:
        if word not in stop_words_set:
            output.append(word)

    return output

import numpy as np
import gensim.downloader as api
w2v_model = api.load('word2vec-google-news-300')


def get_avg_word2vec(tokens, model, vector_size=300):
    vectors = [model[word] for word in tokens if word in model]
    if not vectors:
        return np.zeros(vector_size)
    return np.mean(vectors, axos=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Took help from open source for writing this "X"
X_vectors = df['message'].apply(lambda x: get_avg_word2vec(preprocessing(x), w2v_model))

X = np.vstack(X_vectors.values) 
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, y_pred)}")

def predict_message_class(model, w2v_model, message):
    tokens = preprocessing(message)
    vector = get_avg_word2vec(tokens, w2v_model)
    prediction = model.predict([vector])[0]
    return "spam" if prediction == 1 else "ham"
