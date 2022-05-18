from sklearn import cluster
from gensim.models import word2vec
import numpy as np
import matplotlib.pyplot as plt
#sklearn进行据类分析

model = word2vec.Word2Vec.load('./model_px.model')
names=[]
for line in open("name.txt","r",encoding='utf-8'):
    line = line.strip('\n')
    names.append(line)
names = [name for name in names if name in model.wv]
name_vectors = [model.wv[name] for name in names]

n=4
label = cluster.KMeans(n).fit(name_vectors).labels_
print(label)
print("类别1：")
for i in range(len(label)):
    if label[i]==0:
        print(names[i],end=" ")
print("\n")
print("类别2：")
for i in range(len(label)):
    if label[i]==1:
        print(names[i],end=" ")
print("\n")
print("类别3：")
for i in range(len(label)):
    if label[i]==2:
        print(names[i],end=" ")

print("\n")
print("类别4：")
for i in range(len(label)):
    if label[i]==3:
        print(names[i],end=" ")

'''
print("\n")
print("类别5：")
for i in range(len(label)):
    if label[i]==4:
        print(names[i],end=" ")

print("\n")
print("类别6：")
for i in range(len(label)):
    if label[i]==5:
        print(names[i],end=" ")

'''
