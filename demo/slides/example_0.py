# This model is ~2 gb to download.
import gensim.downloader as api

# Load the pretrained Word2Vec model (Google's Word2Vec trained on Google News)
model = api.load("word2vec-google-news-300")

# Gender analogies
# result = model.most_similar(positive=['king', 'woman'], negative=['man'], topn=1)
# result = model.most_similar(positive=['prince', 'man'], negative=['woman'], topn=1)

# Singular vs plural
# result = model.most_similar(positive=['dog', 'kitten'], negative=['puppy'], topn=1)
# result = model.most_similar(positive=['book', 'pages'], negative=['page'], topn=1)

# Verb tense analogies
# result = model.most_similar(positive=['walk', 'running'], negative=['walking'], topn=1)
# result = model.most_similar(positive=['write', 'singing'], negative=['writing'], topn=1)

# Country capital relationships
# result = model.most_similar(positive=['France', 'Berlin'], negative=['Paris'], topn=1)
# result = model.most_similar(positive=['India', 'Moscow'], negative=['Delhi'], topn=1)

# Superlative & Comparative Forms
# result = model.most_similar(positive=['Big', 'Smaller'], negative=['Bigger'], topn=1)
# result = model.most_similar(positive=['Good', 'Better'], negative=['Worse'], topn=1)

print(result)