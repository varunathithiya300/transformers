import torch
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt

# Define input sentence (3 words, 4-dimensional embeddings)
input_vectors = torch.tensor([
    [1, 0, 1, 0],  # "The"
    [0, 1, 0, 1],  # "cat"
    [1, 1, 1, 1]   # "sat"
], dtype=torch.float32)

# Randomly initialize Query, Key, and Value weight matrices
torch.manual_seed(42)
W_Q = torch.rand(4, 4)
W_K = torch.rand(4, 4)
W_V = torch.rand(4, 4)

# Compute Query, Key, and Value matrices
Q = input_vectors @ W_Q  # (3x4) * (4x4) -> (3x4)
K = input_vectors @ W_K
V = input_vectors @ W_V

# Compute attention scores (Q @ K^T) / sqrt(d_k)
d_k = K.shape[1]  # Dimension of key vectors (4)
scores = (Q @ K.T) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# Apply softmax to get attention weights
attention_weights = F.softmax(scores, dim=1)

# Compute final output by weighting Value (V) with attention scores
output = attention_weights @ V

print("Attention Weights:\n", attention_weights)
print("Final Output:\n", output)
