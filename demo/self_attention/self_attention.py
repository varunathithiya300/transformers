import numpy as np

# Define input sentence (3 words with 4-dimensional embeddings)
input_vectors = np.array([
    [1, 0, 1, 0],  # "The"
    [0, 1, 0, 1],  # "cat"
    [1, 1, 1, 1]   # "sat"
])

# Randomly initialize Query, Key, and Value weight matrices
np.random.seed(42)
W_Q = np.random.rand(4, 4)
W_K = np.random.rand(4, 4)
W_V = np.random.rand(4, 4)

# Compute Query, Key, and Value matrices
Q = np.dot(input_vectors, W_Q)
K = np.dot(input_vectors, W_K)
V = np.dot(input_vectors, W_V)

# Compute attention scores (Q @ K^T) / sqrt(d_k)
d_k = K.shape[1]  # Dimension of key vectors
scores = np.dot(Q, K.T) / np.sqrt(d_k)

# Apply softmax to get attention weights
attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

# Compute final output by weighting Value (V) with attention scores
output = np.dot(attention_weights, V)

print("Attention Weights:\n", attention_weights)
print("Final Output:\n", output)
