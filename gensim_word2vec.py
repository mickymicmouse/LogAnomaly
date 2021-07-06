# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 17:49:20 2021

@author: seoun
"""

#import nltk
#nltk.download('punkt')

import urllib.request
import zipfile
from lxml import etree
import re
import pandas as pd
import pickle
import os

#데이터 로드
Data_root = r"C:\Users\seoun\Desktop\Labs\LogData Project\Embedding\Data"
df_acc_log = pd.read_csv(r"C:\Users\seoun\Desktop\Labs\LogData Project\elasticData.csv", encoding= "utf8")


with open(os.path.join(Data_root, "elasticData"), "wb") as file:
    pickle.dump(df_acc_log, file)

with open(os.path.join(Data_root, "elasticData"),"rb") as file:
    df = pickle.load(file)


uri = df_acc_log['uri']
result = [sentence[1:].split("/") for sentence in uri]

#Word2Vec 학습
from gensim.models import Word2Vec
model = Word2Vec(sentences = result, vector_size= 100, window = 5, min_count=5, workers = 4, sg = 0)
model_result = model.wv.most_similar("ham")
print(model_result)


