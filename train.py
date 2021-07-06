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
lens = [len(i) for i in result]
#가장 긴 uri 
print(max(lens))
# URI 종류 세기 
print(len(set(uri)))

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
from gensim.models import KeyedVectors

model = Word2Vec.load(os.path.join(Data_root, "uri_embedding_matrix"))
print(model.wv['prog'])


import sys
sys.path.append(r'C:\Users\seoun\Desktop\Labs\LogData Project\LogAnomaly')
import lstm



options = dict()
options['data_dir'] = '../data/'
options['window_size'] = 10
options['device'] = "cpu"

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window

# Features
options['sequentials'] = True
options['quantitatives'] = True
options['semantics'] = False
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 100
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 427

# Train
options['batch_size'] = 2048
options['accumulation_step'] = 1

options['optimizer'] = 'adam'
options['lr'] = 0.001
options['max_epoch'] = 370
options['lr_step'] = (300, 350)
options['lr_decay_ratio'] = 0.1

options['resume_path'] = None
options['model_name'] = "loganomaly"
options['save_dir'] = "../result/loganomaly/"

# Predict
options['model_path'] = "../result/loganomaly/loganomaly_epoch299.pth"
options['num_candidates'] = 9


Model = lstm.loganomaly(input_size=options['input_size'],
                   hidden_size=options['hidden_size'],
                   num_layers=options['num_layers'],
                   num_keys=options['num_classes'])

train_logs, train_labels = sliding_window()
valid_logs, valid_labels = sliding_window()



def sliding_window(data_dir, datatype, window_size, sample_ratio=1):
    '''
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []
    if datatype == 'train':
        data_dir += 'hdfs/hdfs_train'
    if datatype == 'val':
        data_dir += 'hdfs/hdfs_test_normal'

    with open(data_dir, 'r') as f:
        for line in f.readlines():
            num_sessions += 1
            line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

            for i in range(len(line) - window_size):
                Sequential_pattern = list(line[i:i + window_size])
                Quantitative_pattern = [0] * 28
                log_counter = Counter(Sequential_pattern)

                for key in log_counter:
                    Quantitative_pattern[key] = log_counter[key]
                Semantic_pattern = []
                for event in Sequential_pattern:
                    if event == 0:
                        Semantic_pattern.append([-1] * 300)
                    else:
                        Semantic_pattern.append(event2semantic_vec[str(event -
                                                                       1)])
                Sequential_pattern = np.array(Sequential_pattern)[:,
                                                                  np.newaxis]
                Quantitative_pattern = np.array(
                    Quantitative_pattern)[:, np.newaxis]
                result_logs['Sequentials'].append(Sequential_pattern)
                result_logs['Quantitatives'].append(Quantitative_pattern)
                result_logs['Semantics'].append(Semantic_pattern)
                labels.append(line[i + window_size])

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels

