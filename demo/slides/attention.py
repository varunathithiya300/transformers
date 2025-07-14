from transformers import BertTokenizer, BertModel
from bertviz.head_view import show  # Import correctly
from bertviz.head_view import visualize
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Encode a sentence
sentence = "The quick brown fox jumps over the lazy dog."
tokens = tokenizer(sentence, return_tensors='pt')

# Get attention scores (disable gradients for efficiency)
with torch.no_grad():
    outputs = model(**tokens)

attentions = outputs.attentions  # Extract attention layers

# Convert token IDs back to readable words (including subwords)
token_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])

# Visualize Attention
show(attentions, token_list)
