For each word in a sentence, self-attention computes:

- **Query (Q)** → What am I looking for?
- **Key (K)** → What is the meaning of this word?
- **Value (V)** → What information does this word carry?

# Attention Score - **(Q.K^T) / sqrt(dk)**

- dk - dimension of the key vectors (used for scaling)
- scores are normalized with softmax -> sum up to 1
- these scores are used to weight the Value (V) vectors -> important words contribute more to the final output

- The scores are normalized with Softmax so they sum to 1.
- These scores are used to weight the Value (V) vectors, so important words contribute more to the final output.
