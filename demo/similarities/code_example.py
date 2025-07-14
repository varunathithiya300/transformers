# Compare Cosine similarity, Euclidean distance, Jaccard Similarity

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import CountVectorizer

# Sample text data
sentence1 = "The cat sat on the mat"
sentence2 = "The dog lay on the rug"

# Convert sentences into vectorized form using bag-of-words (BOW)
vectorizer = CountVectorizer(binary=True)  # Binary to use Jaccard similarity
X = vectorizer.fit_transform([sentence1, sentence2]).toarray()

# Compute Cosine Similarity
cos_sim = cosine_similarity([X[0]], [X[1]])[0][0]

# Compute Euclidean Distance
euc_dist = euclidean_distances([X[0]], [X[1]])[0][0]

# Compute Jaccard Similarity
intersection = np.logical_and(X[0], X[1]).sum()
union = np.logical_or(X[0], X[1]).sum()
jaccard_sim = intersection / union

print(f"Cosine Similarity: {cos_sim:.4f}")
print(f"Euclidean Distance: {euc_dist:.4f}")
print(f"Jaccard Similarity: {jaccard_sim:.4f}")

# -----
#### Comparing Cosine Similarity Using Word Embeddings (Word2Vec & BERT)
### Compute Sentence Similarity Using Word2Vec (GloVe Embeddings)

import numpy as np
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained Word2Vec model (GloVe embeddings)
word_vectors = api.load("glove-wiki-gigaword-50")  # 50-dimensional embeddings

# Function to get sentence embedding (average of word embeddings)
def sentence_embedding(sentence, model):
    words = sentence.lower().split()
    word_vecs = [model[word] for word in words if word in model]
    return np.mean(word_vecs, axis=0) if word_vecs else np.zeros(50)

# Compute sentence embeddings
vec1 = sentence_embedding(sentence1, word_vectors)
vec2 = sentence_embedding(sentence2, word_vectors)

# Compute Cosine Similarity
cos_sim_glove = cosine_similarity([vec1], [vec2])[0][0]

print(f"Cosine Similarity (Word2Vec/GloVe): {cos_sim_glove:.4f}")

# -----
### Compute Sentence Similarity Using BERT Sentence Embeddings

from sentence_transformers import SentenceTransformer

# Load pre-trained BERT model for sentence embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
emb1 = model.encode(sentence1, convert_to_numpy=True)
emb2 = model.encode(sentence2, convert_to_numpy=True)

# Compute Cosine Similarity
cos_sim_bert = cosine_similarity([emb1], [emb2])[0][0]

print(f"Cosine Similarity (BERT): {cos_sim_bert:.4f}")
