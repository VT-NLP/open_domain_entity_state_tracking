from sentence_transformers import util

import torch

relation_embedding=torch.rand([3,10])
question_embedding=torch.torch.LongTensor([0.1,0.3,0.3])

question_embedding = question_embedding.repeat(relation_embedding.size(0),1)
# print(question_embedding)
cos_sim = util.pytorch_cos_sim(question_embedding, relation_embedding)
print(cos_sim)