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
Data_root = r"C:\Users\seoun\Desktop\Labs\LogData Project\LogAnomaly\Data"
with open(os.path.join(Data_root, "elasticData"),"rb") as file:
    df = pickle.load(file)

#%% 데이터 분할


def remove_unknown(Data_root, df_acc_log):
    # Unknown 데이터 삭제처리 
    new_df_acc_log=[]
    for i in range(len(df_acc_log)):
        if i % 5000 ==0:
            print(i)
        dept_code = df_acc_log.iloc[i]['dept_code']
        if dept_code != "UNKNOWN":
            new_df_acc_log.append(df_acc_log.iloc[i])
    new_df = pd.DataFrame(new_df_acc_log)
    
    with open(os.path.join(Data_root, "known_df"), "wb") as file:
        pickle.dump(new_df, file)
    
    return new_df
        

def sorting(Data_root, log_entry):
    # log_entry 별 정렬 시행 
    with open(os.path.join(Data_root, "known_df"),"rb") as file:
        df = pickle.load(file)
    
    # df = new_df.copy()
    
    df.sort_values(by=[log_entry, 'str_time'], inplace = True)
    df.reset_index(drop=True, inplace=True)
    with open(os.path.join(Data_root, "sorted_df_acc_log_"+log_entry), "wb") as file:
        pickle.dump(df, file)
        
    with open(os.path.join(Data_root, "sorted_df_acc_log_"+log_entry), "rb") as file:
        df = pickle.load(file)
    print(log_entry +"별 정렬")
    return df



log_entry= "dept_code"
known_df = remove_unknown(Data_root, df)
df = sorting(Data_root, log_entry) # dept_code, dept_name, user_sn, position_code, position_name



#%% 2차원 그래프 그리기 
def plot_2d_graph(vocabs, xs, ys):
    plt.figure(figsize = (8,6))
    plt.scatter(xs,ys,marker='o')
    for i,v in enumerate(vocabs):
        plt.annotate(v,xy=(xs[i],ys[i]))

#%% 문장 생성 

log_entry = "dept_code"
with open(os.path.join(Data_root, "sorted_df_acc_log_"+log_entry), "rb") as file:
    df = pickle.load(file)

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

#%% log_entry별로 묶은 리스트 생성 

def listing(Data_root, log_entry):
    # log_entry 별로 URI 리스트화 
    with open(os.path.join(Data_root, "sorted_df_acc_log_"+log_entry), "rb") as file:
        df = pickle.load(file)
    
    dept_name = df[log_entry].unique()
    dept_n = len(df[log_entry].unique())
    dept_dict=dict()
    for i in range(dept_n):
        dept_dict[dept_name[i]]=[]
    
    for idx in range(len(df)):
        if idx%5000 == 0:
            print("%d 번째 로그입니다" %idx)
        dept_dict[df[log_entry][idx]].append(df.iloc[idx]['uri'])
    
    uri_seq = []
    for k in dept_dict.keys():
        uri_seq.append(dept_dict[k])
    
    uri_seq.sort(key = len, reverse = True)
    with open(os.path.join(Data_root, "uri_seq_"+log_entry), "wb") as file:
        pickle.dump(uri_seq, file)
    return uri_seq


def train_valid_split(Data_root, log_entry, total_len):
    # Train과 Valid 분할 
    with open(os.path.join(Data_root, "uri_seq_"+log_entry), "rb") as file:
        uri_seq = pickle.load(file)
    
    nums = total_len
    ratio = 0.8
    train_num = int(nums*ratio)
    
    # train과 valid의 개수를 비슷하게 맞추기 위함 
    count = 0
    idx = 0
    for seq in uri_seq:
        idx += 1
        count += len(seq)
        if count>=train_num:
            break
    
    uri_train = uri_seq[:idx]
    uri_valid = uri_seq[idx:]
    
    with open(os.path.join(Data_root, "uri_train_"+log_entry), "wb") as file:
        pickle.dump(uri_train, file)
    
    with open(os.path.join(Data_root, "uri_valid_"+log_entry), "wb") as file:
        pickle.dump(uri_valid, file)
        
    return uri_train, uri_valid

log_entry = "dept_code"
uri_seq = listing(Data_root, log_entry)
trainset, validset = train_valid_split(Data_root, log_entry, len(df))


#%% Embedding 모델 로드 후 OPTION 설정

import torch
import torch.nn as n
from gensim.models import KeyedVectors

model = Word2Vec.load(os.path.join(Data_root, "uri_embedding_matrix"))
print(model.wv['prog'])


import sys
sys.path.append(r'C:\Users\seoun\Desktop\Labs\LogData Project\LogAnomaly')
import lstm



options = dict()
options['data_dir'] = r'C:\Users\seoun\Desktop\Labs\LogData Project\LogAnomaly\Data'
options['window_size'] = 10
options['device'] = "cpu"

# Smaple
options['sample'] = "sliding_window"
options['window_size'] = 10  # if fix_window

# Features
options['sequentials'] = False
options['quantitatives'] = True
options['semantics'] = True
options['feature_num'] = sum(
    [options['sequentials'], options['quantitatives'], options['semantics']])

# Model
options['input_size'] = 100
options['hidden_size'] = 64
options['num_layers'] = 2
options['num_classes'] = 237

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
options['save_dir'] = os.path.join(Data_root,"uri_result")

# Predict
options['model_path'] = r"C:\Users\seoun\Desktop\Labs\LogData Project\LogAnomaly\Data\uri_result\loganomaly_last.pth"
options['num_candidates'] = 9



#%% LSTM 모델 로드 

Model = lstm.loganomaly(input_size=options['input_size'],
                   hidden_size=options['hidden_size'],
                   num_layers=options['num_layers'],
                   num_keys=options['num_classes'])


#%% 데이터 준비 
import json
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

def sliding_window(data_dir, datatype,log_entry, window_size ):
    '''
    데이터 window size 에 맞게 생성 
    dataset structure
        result_logs(dict):
            result_logs['feature0'] = list()
            result_logs['feature1'] = list()
            ...
        labels(list)
    '''
    
    with open(os.path.join(Data_root, "sorted_df_acc_log_"+log_entry), "rb") as file:
        df = pickle.load(file)
    uri_n = len(df['uri'].value_counts())
    dfvc = df['uri'].value_counts()
    uri_unique = df['uri'].unique()
    
    uri_dict = dict()
    for i in range(uri_n):
        uri_dict[uri_unique[i]]=i
    uri_dict["OOV"]=uri_n
    
    
    model = Word2Vec.load(os.path.join(data_dir, "uri_embedding_matrix"))
    # event2semantic_vec = read_json(data_dir + 'hdfs/event2semantic_vec.json')
    num_sessions = 0
    result_logs = {}
    result_logs['Sequentials'] = []
    result_logs['Quantitatives'] = []
    result_logs['Semantics'] = []
    labels = []
    if datatype == 'train':
        data_dir = os.path.join(data_dir,'uri_train_'+log_entry)
    if datatype == 'valid':
        data_dir = os.path.join(data_dir,'uri_valid_'+log_entry)
    
    
    with open(data_dir, 'rb') as f:
        uri_train = pickle.load(f)
    for line in tqdm(uri_train):
        
        num_sessions += 1
        # line = tuple(map(lambda n: n - 1, map(int, line.strip().split())))

        for i in range(len(line) - window_size):
            Sequential_pattern = list(line[i:i + window_size])
            Quantitative_pattern = [0] * (uri_n+1) # uri 종류 개수 
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[uri_dict[key]] = log_counter[key]
            Semantic_pattern = []
            for event in Sequential_pattern:
                segment = event[1:].split("/")
                for part_idx in range(6):
                    if part_idx>=len(segment):
                        Semantic_pattern.append([0] * 100) # 부족한 부분 zero padding
                    else:
                        if segment[part_idx] not in model.wv.index_to_key:
                            Semantic_pattern.append([0] * 100) # OOV 처리
                        else:    
                            Semantic_pattern.append(model.wv[segment[part_idx]])
                
                
            Sequential_pattern = np.array(Sequential_pattern)[:,np.newaxis]
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
            result_logs['Sequentials'].append(Sequential_pattern)
            result_logs['Quantitatives'].append(Quantitative_pattern)
            result_logs['Semantics'].append(Semantic_pattern)
            labels.append(uri_dict[line[i + window_size]])


    print('File {}, number of sessions {}'.format(data_dir, num_sessions))
    print('File {}, number of seqs {}'.format(data_dir,
                                              len(result_logs['Sequentials'])))

    return result_logs, labels


train_logs, train_labels = sliding_window(Data_root, 'train',log_entry, 10)
valid_logs, valid_labels = sliding_window(Data_root, 'valid', log_entry,10)


#%% 데이터 클래스화 

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class log_dataset(Dataset):
    def __init__(self, logs, labels, seq=True, quan=False, sem=False):
        self.seq = seq
        self.quan = quan
        self.sem = sem
        if self.seq:
            self.Sequentials = logs['Sequentials']
        if self.quan:
            self.Quantitatives = logs['Quantitatives']
        if self.sem:
            self.Semantics = logs['Semantics']
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        if self.seq:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.float)
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                dtype=torch.float)
        if self.sem:
            log['Semantics'] = torch.tensor(self.Semantics[idx],
                                            dtype=torch.float)
        return log, self.labels[idx]


train_dataset = log_dataset(logs=train_logs,
                                labels=train_labels,
                                seq=options["sequentials"],
                                quan=options["quantitatives"],
                                sem=options["semantics"])
valid_dataset = log_dataset(logs=valid_logs,
                                labels=valid_labels,
                                seq=options["sequentials"],
                                quan=options["quantitatives"],
                                sem=options["semantics"])

print(train_dataset[0])


#%% 데이터 로딩
import torch.nn as nn
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset,
                               batch_size=options['batch_size'],
                               shuffle=True,
                               pin_memory=True)
valid_loader = DataLoader(valid_dataset,
                               batch_size=options['batch_size'],
                               shuffle=False,
                               pin_memory=True)


num_train_log = len(train_dataset)
num_valid_log = len(valid_dataset)

print('Find %d train logs, %d validation logs' %
      (num_train_log, num_valid_log))
print('Train batch size %d ,Validation batch size %d' %
      (options['batch_size'], options['batch_size']))

#%% 모델세팅
Model = Model.to(options['device'])
if options['optimizer'] == 'sgd':
    optimizer = torch.optim.SGD(Model.parameters(),
                                     lr=options['lr'],
                                     momentum=0.9)
elif options['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(
        Model.parameters(),
        lr=options['lr'],
        betas=(0.9, 0.999),
    )



def save_checkpoint(epoch, model, best_loss, log, best_score,optimizer, save_dir,model_name, save_optimizer=True, suffix=""):
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "best_loss": best_loss,
        "log": log,
        "best_score": best_score
    }
    if save_optimizer:
        checkpoint['optimizer'] = optimizer.state_dict()
    save_path = os.path.join(save_dir ,model_name + "_" + suffix + ".pth")
    torch.save(checkpoint, save_path)
    print("Save model checkpoint at {}".format(save_path))


def save_parameters(options, filename):
    with open(filename, "w+") as f:
        for key in options.keys():
            f.write("{}: {}\n".format(key, options[key]))
            
def save_log(log, save_dir):
    try:
        for key, values in log.items():
            pd.DataFrame(values).to_csv(os.path.join(save_dir, key + "_log.csv"),
                                        index=False)
        print("Log saved")
    except:
        print("Failed to save logs")

start_epoch = 0
best_loss = 1e10
best_score = -1
max_epoch = options['max_epoch']
save_parameters(options, options['save_dir']+"parameters.txt")
log = {
            "train": {key: []
                      for key in ["epoch", "lr", "time", "loss"]},
            "valid": {key: []
                      for key in ["epoch", "lr", "time", "loss"]}
        }

if options['resume_path'] is not None:
    if os.path.isfile(options['resume_path']):
        print("Resuming from {}".format(options['resume_path']))
        checkpoint = torch.load(options['resume_path'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        log = checkpoint['log']
        best_f1_score = checkpoint['best_f1_score']
        model.load_state_dict(checkpoint['state_dict'])
        if "optimizer" in checkpoint.keys():
            print("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Checkpoint not found")


#%% 모델 학습 
import time

for epoch in range(start_epoch, max_epoch):
    if epoch == 0:
        optimizer.param_groups[0]['lr'] /= 32
    if epoch in [1,2,3,4,5]:
        optimizer.param_groups[0]['lr'] *= 2
    if epoch in options['lr_step']:
        optimizer.param_groups[0]['lr'] *= options['lr_decay_ratio']
    # Training
    log['train']['epoch'].append(epoch)
    start = time.strftime("%H:%M:%S")
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    print("Starting epoch: %d | phase: train | ⏰: %s | Learning rate: %f" %
          (epoch, start, lr))
    log['train']['lr'].append(lr)
    log['train']['time'].append(start)
    Model.train()
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    tbar = tqdm(train_loader, desc="\r")
    num_batch = len(train_loader)
    total_losses = 0
    for i, (ln, label) in enumerate(tbar):
        features = []
        for value in ln.values():
            features.append(value.clone().detach().to(options['device']))
        output = Model(features=features, device=options['device'])
        loss = criterion(output, label.to(options['device']))
        total_losses += float(loss)
        loss /= options['accumulation_step'] 
        loss.backward()
        if (i + 1) % options['accumulation_step']  == 0:
            optimizer.step()
            optimizer.zero_grad()
        tbar.set_description("Train loss: %.5f" % (total_losses / (i + 1)))

    log['train']['loss'].append(total_losses / num_batch)
    # End of Training Code
    
    if epoch >=max_epoch // 2 and epoch % 2==0:
        # validation
        Model.eval()
        log['valid']['epoch'].append(epoch)
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        log['valid']['lr'].append(lr)
        start = time.strftime("%H:%M:%S")
        print("Starting epoch: %d | phase: valid | ⏰: %s " % (epoch, start))
        log['valid']['time'].append(start)
        total_losses = 0
        criterion = nn.CrossEntropyLoss()
        tbar = tqdm(valid_loader, desc="\r")
        num_batch = len(valid_loader)
        for i, (log, label) in enumerate(tbar):
            with torch.no_grad():
                features = []
                for value in log.values():
                    features.append(value.clone().detach().to(options['device']))
                output = Model(features=features, device=options['device'])
                loss = criterion(output, label.to(options['device']))
                total_losses += float(loss)
        print("Validation loss:", total_losses / num_batch)
        log['valid']['loss'].append(total_losses / num_batch)

        if total_losses / num_batch < best_loss:
            best_loss = total_losses / num_batch
            save_checkpoint(epoch,
                                 save_optimizer=False,
                                 suffix="bestloss")
        
        # End of Validation Code
        
        save_checkpoint(epoch, Model, best_loss, log, best_score, optimizer, options['save_dir'], options['model_name'])
    save_checkpoint(epoch, Model, best_loss, log, best_score, optimizer, options['save_dir'], options['model_name'], suffix = "last")
    save_log(log, options['save_dir'])
    


log['train']['epoch'].append(epoch)
start = time.strftime("%H:%M:%S")


#%% 예측 결과값 (Validation)


num_candidates = options['num_candidates']
num_classes = options['num_classes']
input_size = options['input_size']
sequentials = options['sequentials']
quantitatives = options['quantitatives']
semantics = options['semantics']
batch_size = options['batch_size']


Model = Model.to(options['device'])
Model.load_state_dict(torch.load(options['model_path'])['state_dict'])
Model.eval()
print('model_path: {}'.format(options['model_path']))
# test_normal_loader, test_normal_length = generate('hdfs_test_normal')


TP = 0
FP = 0
# Test the model
start_time = time.time()
with torch.no_grad():
    for line in tqdm(validset):
        for i in range(len(line) - options['window_size']):
            seq0 = line[i:i + options['window_size']]
            label = line[i + options['window_size']]
            
            seq1 = [0] * 238
            log_conuter = Counter(seq0)
            for key in log_conuter:
                seq1[key] = log_conuter[key]

            seq0 = torch.tensor(seq0, dtype=torch.float).view(
                -1, options['window_size'], input_size).to(options['device'])
            seq1 = torch.tensor(seq1, dtype=torch.float).view(
                -1, num_classes, input_size).to(options['device'])
            label = torch.tensor(label).view(-1).to(options['device'])
            output = model(features=[seq0, seq1], device=options['device'])
            predicted = torch.argsort(output,
                                      1)[0][-num_candidates:]
            if label not in predicted:
                FP += valid_loader[line]
                break
            else:
                TP += valid_loader[line]
                

# Compute precision, recall and F1-measure
# FN = test_abnormal_length - TP
# P = 100 * TP / (TP + FP)
# R = 100 * TP / (TP + FN)
Acc = 100 * TP / (TP + FP)
# F1 = 2 * P * R / (P + R)
"""
print(
    'false positive (FP): {}, false negative (FN): {}, Precision: {:.3f}%, Recall: {:.3f}%, F1-measure: {:.3f}%'
    .format(FP, FN, P, R, F1))
"""
print('Finished Predicting')
elapsed_time = time.time() - start_time
print('ACC: {}, TP: {}, FP: {}'.format(Acc, TP, FP))
print('elapsed_time: {}'.format(elapsed_time))







