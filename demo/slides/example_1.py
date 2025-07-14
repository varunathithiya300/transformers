from sentence_transformers import SentenceTransformer, util

model1 = SentenceTransformer('all-MiniLM-L6-v2')
model2 = SentenceTransformer('all-mpnet-base-v2')

# sentences = ["Football is my favourite sport","Football is not my favourite sport"]
# sentences = ["football", "golfball"]
sentences = ["How to cook pasta ?", "Easy spaghetti recipe"]
# sentences = ["moto gp", "formula one"]
# sentences = ["bert is the foundation of sentence transformer", "sentence transformer is a library built on bert"]

embeddings1 = model1.encode(sentences)
embeddings2 = model2.encode(sentences)

similarity1 = util.cos_sim(embeddings1[0], embeddings1[1])
print(f"all-MiniLM-L6-v2: {similarity1.item():.2f}", ";", "Dim :", embeddings1.shape)

similarity2 = util.cos_sim(embeddings2[0], embeddings2[1])
print(f"all-mpnet-base-v2: {similarity2.item():.2f}",";", "Dim :", embeddings2.shape)