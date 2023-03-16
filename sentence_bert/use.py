from sentence_transformers import SentenceTransformer, util

# model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
# sentences = ['Lack of saneness',
#              'Absence of sanity',
#              'A man is eating food.',
#              'A man is eating a piece of bread.',
#              'The girl is carrying a baby.',
#              'A man is riding a horse.',
#              'A woman is playing violin.',
#              'Two men pushed carts through the woods.',
#              'A man is riding a white horse on an enclosed ground.',
#              'A monkey is playing drums.',
#              'A cheetah is running behind its prey.']
# sentence_embeddings = model.encode(sentences)
#
# for sentence, embedding in zip(sentences, sentence_embeddings):
#     print("Sentence:", sentence)
#     print("Embedding:", embedding)
#     print(len(embedding))
#     print("")

# model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))