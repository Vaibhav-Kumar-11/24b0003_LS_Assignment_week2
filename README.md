# 24b0003_LS_Assignment_week2

# 📩 SMS Spam Classification using Word2Vec & Logistic Regression

## 📌 Problem Statement

Build a machine learning model that classifies SMS messages as **"spam"** or **"ham"** (non-spam), using the **SMS Spam Collection Dataset**.

---

## 🧠 Approach

The task was completed using the following step-by-step NLP + ML pipeline:

### 1. **Data Loading**
- Used `pandas` to load the dataset (`spam.csv`) with columns:
  - `label`: spam/ham
  - `message`: the SMS text

### 2. **Text Preprocessing**
Performed basic preprocessing to prepare the text data for vectorization:
- Converted to lowercase
- Tokenized using `split()` (fallback due to `nltk` environment issues)
- Removed stopwords using `nltk.corpus.stopwords`
> _Note_: Tokenization was kept simple for this assignment. Planned to upgrade to `nltk.word_tokenize()` with punctuation filtering in later iterations.

### 3. **Word2Vec Embeddings**
Used the pre-trained **Google News Word2Vec model (300 dimensions)** from `gensim.downloader`:
- For each SMS message, calculated the average word vector of all known words in the message.

### 4. **Model Training**
- Transformed all messages into vector format using the above function.
- Split the dataset into **80% training** and **20% testing** using `train_test_split`.
- Trained a **Logistic Regression classifier** from `sklearn.linear_model`.

### 5. **Evaluation**
- Printed **accuracy score** on the test set.
- Wrote a custom function `predict_message_class()` to classify any new message.

---

## 🔢 Sample Output

```bash
Accuracy : 0.96

Input: "Congratulations! You've won a free cruise. Call now!"
Prediction: spam

Input: "Hey bro, I’ll call you in 5 minutes."
Prediction: ham


📊 Libraries & Tools Used
Library	                   Purpose
pandas	          Data loading & manipulation
nltk	          Stopwords filtering
gensim	          Pre-trained Word2Vec embeddings
scikit-learn	  ML model, train/test split, metrics
numpy	          Array/vector handling


🧪 Learning Outcomes:
1.Learned how to convert raw text into word embeddings

2.Understood the importance of preprocessing and cleaning

3.Realized how pre-trained models (Word2Vec) simplify feature extraction

4.Practiced writing custom prediction wrappers for reuse

5.Faced and debugged real-world errors (nltk tokenizer issues, environment path conflicts)


🙋 Author
Vaibhav Kumar
IIT Bombay – Learners’ Space ML Assignment Week 2
Submitted as part of SoS'25 program

