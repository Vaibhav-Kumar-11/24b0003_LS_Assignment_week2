import math

corpus = [
    'the sun is a star',
    'the moon is a satellite',
    'the sun and moon are celestial bodies'
]

tf = []
for line in corpus:
    line = line.lower()  # If we are considering a general case so there might be chances ki the sentence is something like "The ......the" and we would end up treating them as different.
    individual_words = line.split()

    frequency_dict = {}

    for word in individual_words:
        if word in frequency_dict: 
            frequency_dict[word] += 1
        else:
            frequency_dict[word] = 1    

    tf.append(frequency_dict) 


# L = number of lines that contain that particular word
# idf = log(len(corpus)/L)  

# As we know ki set contains only unique words, So:
# Took some ideational part from open source after this point.

unique_words = set()
for doc in tf:
    for word in doc:
        unique_words.add(word)

idf = {}

for word in unique_words:
    doc_count = 0
    for doc in tf:
        if word in doc:
            doc_count+=1
    idf[word] = math.log(len(corpus)/doc_count)

tfidf = []

for doc in tf:
    tfidf_doc = {}
    for word in doc:
        tfidf_doc[word] = doc[word] * idf[word]
    tfidf.append(tfidf_doc)

print(tfidf)