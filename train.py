# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 14:20:29 2021

@author: seoun
"""

import urllib.request
import zipfile
from lxml import etree
import re
import pandas as pd
import pickle
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import os


#데이터 로드
Data_root = r"C:\Users\seoun\Desktop\Labs\LogData Project\Embedding\Data"
with open(os.path.join(Data_root, "elasticData"),"rb") as file:
    df = pickle.load(file)
    

#%% 2차원 그래프 그리기 
def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize = (8,6))
    plt.scatter(xs,ys,marker='o')
    for i,v in enumerate(vocabs):
        plt.annotate(v,xy=(xs[i],ys[i]))

#%% 문장 생성 
uri = df['uri']
result = [sentence[1:].split("/") for sentence in uri]
sentences = result



#%% 문장 이용해서 단어 벡터 생성 
model = Word2Vec(sentences, vector_size= 100, window = 3, min_count=0, workers = 1, sg = 0)
word_vectors = model.wv
# 모델 저장 
model.save(os.path.join(Data_root,'uri_embedding_matrix'))

vocabs = list(word_vectors.index_to_key)
word_vectors_list = [word_vectors[v] for v in vocabs]

#유사도 측정 
print(word_vectors.similarity(w1='prog', w2 = 'view.do'))

#%% 단어 PCA 시각화  
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xys = pca.fit_transform(word_vectors_list)
xs = xys[:,0]
ys = xys[:,1]
plot_2d_graph(vocabs, xs, ys)



#%% 모델 로드 후 Train 

import torch
import torch.nn as n
from torchtext.vocab import Vectors


model = Word2Vec.load(os.path.join(Data_root, "uri_embedding_matrix"))
print(model.wv['prog'])

