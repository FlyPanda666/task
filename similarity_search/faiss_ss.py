import faiss
import numpy as np


def construct_index_repo():
    pass


d = 64                                           # 向量维度
nb = 100000                                      # index向量库的数据量
nq = 10000                                       # 待检索query的数目
np.random.seed(1234)
xb = np.random.random((nb, d)).astype('float32')
print(xb)
xb[:, 0] += np.arange(nb) / 1000.                # index向量库的向量
print(xb)
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.                # 待检索的query向量


index = faiss.IndexFlatL2(d)
print(index.is_trained)         # 输出为True，代表该类index不需要训练，只需要add向量进去即可
index.add(xb)                   # 将向量库中的向量加入到index中
print(index.ntotal)             # 输出index中包含的向量总数，为100000

k = 4                     # topK的K值
D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
print(I[:5])
print(D[-5:])


def construct_index(xb, top_k):
    dim, measure = 64, faiss.METRIC_L2
    param = 'Flat'
    index = faiss.index_factory(dim, param, measure)
    print(index.is_trained)                                   # 输出为True
    print(index.ntotal)             # 输出index中包含的向量总数，为100000

    index.add(xb)                                      # 向index中添加向量
    return index


index = construct_index(xb)
D, I = index.search(xq, k)

print(D)
print(I)
