from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

cv_model = CountVectorizer()
cv_model_transform = cv_model.fit_transform(corpus)

print(cv_model_transform.toarray())   # Important how to call that (remember)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_model = TfidfVectorizer()
tfidf_model_transform = tfidf_model.fit_transform(corpus)

print(tfidf_model_transform.toarray())