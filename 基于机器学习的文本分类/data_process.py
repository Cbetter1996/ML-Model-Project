#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 14:15:56 2023

@author: mac
"""

import pandas as pd
import numpy as np
import tensorflow as tf


from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##from sklearn.externals import joblib
import joblib
import jieba



dataset_path = '/Users/mac/Desktop/person file/project_train/Text_Classification_Based_on_Machine_Learning/Data/cnews_train.txt'
stopwords_path = '/Users/mac/Desktop/person file/project_train/Text_Classification_Based_on_Machine_Learning/Data/cnews_vocab.txt'
save_path = '/Users/mac/Desktop/person file/project_train/Text_Classification_Based_on_Machine_Learning/Data/save_category.txt'
model_save_path = '/Users/mac/Desktop/person file/project_train/Text_Classification_Based_on_Machine_Learning/Data/model_save.pkl'


##读取数据
def read_data(dataset_path, stopwords_path):
    stopwords = list()
    with open(dataset_path, encoding='utf-8') as f1:
        data = f1.readlines()
    with open(stopwords_path, encoding='utf-8') as f2:
        temp_stopwords = f2.readlines()
    for word in temp_stopwords:
        stopwords.append(word[:-1])
    return data, stopwords, temp_stopwords


data, stopwords, temp_stopwords = read_data(dataset_path, stopwords_path)




##将文本的类别写到本地
def save_categories(data, save_path):
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('|'.join(data))
        



##数据预处理, 将数据进行标准化清洗，并生成train和test两类
def pre_data(data, stopwords, test_size=0.2):
    label_list = list()
    text_list = list()
    for line in data:
        label, text = line.split('\t', 1)
        # print(text)
        # print(label)
        seg_text = [word for word in jieba.cut(text) if word not in stopwords]
        text_list.append(' '.join(seg_text))
        label_list.append(label)
    # 标签转化为one-hot格式
    encoder_nums = LabelEncoder()
    label_nums = encoder_nums.fit_transform(label_list)
    categories = list(encoder_nums.classes_)
    save_categories(categories, save_path)
    label_nums = np.array([label_nums]).T
    encoder_one_hot = OneHotEncoder()
    label_one_hot = encoder_one_hot.fit_transform(label_nums)
    label_one_hot = label_one_hot.toarray()
    return model_selection.train_test_split(text_list, label_one_hot, test_size=test_size, random_state=1024)


X_train, X_test, y_train, y_test = pre_data(data, stopwords, test_size=0.2)


##X的词语向量化 
def get_tfidf(X_train, X_test):
    vectorizer = TfidfVectorizer(min_df=100)
    vectorizer.fit_transform(X_train)
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer



##数据清洗函数组装
def provide_data():
    data, stopwords = read_data()
    #  1、提取bag of word参数
    #  2、提取tf-idf特征参数
    X_train, X_test, y_train, y_test = pre_data(data, stopwords, test_size=0.2)
    X_train_vec, X_test_vec, vectorizer = get_tfidf(X_train, X_test)
    joblib.dump(vectorizer, model_save_path)
    #  3、提取word2vec特征参数
    return X_train_vec, X_test_vec, y_train, y_test



##数据shuffle处理，再将数据分批传给模型
def batch_iter(x, y, batch_size=64):
    data_len = len(x)
    num_batch = int((data_len-1)/batch_size)+1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    for i in range(num_batch):
        start_id = i*batch_size
        end_id = min((i+1)*batch_size, data_len)
        yield x_shuffle[start_id: end_id], y_shuffle[start_id: end_id]











