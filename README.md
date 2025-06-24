# 24b0003_LS_Assignment_week2

# 📚 Text Vectorization & Spam Classification – ML Week 2

Hey! I'm **Vaibhav (24b0003)** and this repository contains my Week 2 submissions for Learners' Space ML track.  
This week focused on mastering **text vectorization techniques** through:
- ✍️ Manual TF-IDF implementation
- 🔧 Built-in vectorizers from `scikit-learn`
- 📩 SMS Spam classification using **Word2Vec** + Logistic Regression

---

## 📌 1. Manual TF-IDF Vectorization 🧠

### 📄 Corpus:
```python
corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]
🛠️ What I Did:
Converted text to lowercase for uniformity

Created term frequency (TF) dictionaries

Calculated inverse document frequency (IDF) manually

Combined to get TF-IDF vectors per document

🔍 Key Concepts:
TF = frequency of term in the document

IDF = log(total docs / docs containing the term)

Result: A list of TF-IDF dictionaries per document

🧪 Sample Output:
[{'the': 0.0, 'sun': 0.405, ...}, {'moon': 0.405, ...}, ...]


📌 2. Text Vectorization using Scikit-learn 🧰
🧾 Tools Used:
CountVectorizer – converts text to bag-of-words frequency matrix

TfidfVectorizer – builds normalized TF-IDF vectors automatically

🧪 Output Examples:
python
Copy
Edit
CountVectorizer:
[[1 1 0 0 1 0 1] ...]

TfidfVectorizer:
[[0.5 0.5 0.0 0.0 0.5 ...] ...]
✅ Learning Outcome:
Quick and scalable vectorization using built-in tools, but now with better intuition thanks to the manual version first.



📌 3. SMS Spam Classification using Word2Vec + Logistic Regression 🧠📩
📂 Dataset:
📄 spam.csv (read using pandas)
Columns: label (spam/ham), message (text)

🔍 Preprocessing:
Converted to lowercase

Tokenized and removed stopwords (nltk.corpus.stopwords)

Used pre-trained Google Word2Vec model (300d) via gensim

🧠 Vectorization Logic:
python
Copy
Edit
def get_avg_word2vec(tokens, model, vector_size=300):
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)
📊 Model:
Logistic Regression (sklearn.linear_model)

Train/test split: 80-20, random_state=42

Accuracy printed post evaluation

🧪 Sample Output:
bash
Copy
Edit
Accuracy : 0.9728
🔮 Prediction Function:
python
Copy
Edit
def predict_message_class(model, w2v_model, message):
    tokens = preprocessing(message)
    vector = get_avg_word2vec(tokens, w2v_model)
    return "spam" if model.predict([vector])[0] == 1 else "ham"

🧠 Reflections & Notes
Took partial help from open-source examples for better clarity and direction.

Some design choices (like split() vs nltk.tokenize) were made to avoid library issues during execution.

Will revisit advanced tokenization methods in Week 3.

✅ Summary of Learnings
✅ Built strong intuition behind TF, IDF, and their product

✅ Hands-on with CountVectorizer, TfidfVectorizer

✅ Integrated NLP + ML for a real-world spam classification task

📁 Folder Structure (Recommended)
Copy
Edit
Week2/
├── manual_tfidf.py
├── sklearn_vectorizers.py
├── sms_spam_classifier.py
├── spam.csv
└── README.md
🔗 Dependencies
Python ≥ 3.8

Libraries:

numpy, pandas

nltk

gensim

scikit-learn

📌 Author
Vaibhav Kumar (24b0003)
ML Learner @ Learners' Space Summer 2025
GitHub: Vaibhav-Kumar-11
