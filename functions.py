#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:44:10 2024

@author: clmonter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

## Functions
def normalize(x):
    scaler = StandardScaler()
    z = scaler.fit_transform(x)
    return z

###########################################
## LOAD AND PREPARE DATA
###########################################

def load_variable(full_datos_path, labels_path, id_fold, variable, num_components):

    etiquetas_path = f'{labels_path}/fold_{id_fold}_new/test_chunk_fold_{id_fold}.csv'

    dt_labels = pd.read_csv(etiquetas_path)
    chunks_names = dt_labels['audio_name_chunk'].tolist()

    ## LABELS
    if variable == 'labels':
        data = dt_labels['new_loc_num'].astype(int)

    ## NUMERICAL
    elif variable == 'librosa':
        dt = pd.read_csv(full_datos_path+'/dt_librosa.csv')
   
    elif variable == 'timbral':
        data = pd.read_csv(full_datos_path+'/dt_timbrals.csv')

    elif variable == 'hog':
        data = pd.read_csv(full_datos_path+'/dt_hog.csv')

    elif variable == 'lbp':
        data = pd.read_csv(full_datos_path+'/dt_lbp.csv')

    ## TEXT
    ####################### TF-IDF
    ### YAMNet
    elif variable == 'tfidf_yamnet':
        try:
            data = pd.read_csv(full_datos_path+'/tfidf_yamnet.csv')
        except: 
            data = pd.read_csv(full_datos_path+'/yamnet_tfidf.csv')

    elif variable == 'tfidf_yamnet_02_umbral':
        try:
            data = pd.read_csv(full_datos_path+'/tfidf_yamnet_02_umbral.csv')
        except: 
            data = pd.read_csv(full_datos_path+'/yamnet_tfidf_02_umbral.csv')

    elif variable == 'tfidf_yamnet_03_umbral':
        try:
            data = pd.read_csv(full_datos_path+'/tfidf_yamnet_03_umbral.csv')
        except: 
            data = pd.read_csv(full_datos_path+'/yamnet_tfidf_03_umbral.csv')

    elif variable == 'tfidf_yamnet_redondeo_x10':
        try:
            data = pd.read_csv(full_datos_path+'/tfidf_yamnet_redondeo_x10.csv')
        except: 
            data = pd.read_csv(full_datos_path+'/yamnet_tfidf_redondeo_x10.csv')

    #### PANNs
    elif variable == 'tfidf_panns':
        try:
            data = pd.read_csv(full_datos_path+'/tfidf_panns.csv')
        except: 
            data = pd.read_csv(full_datos_path+'/panns_tfidf.csv')

    elif variable == 'tfidf_panns_02_umbral':
        try:
            data = pd.read_csv(full_datos_path+'/tfidf_panns_02_umbral.csv')
        except: 
            data = pd.read_csv(full_datos_path+'/panns_tfidf_02_umbral.csv')

    elif variable == 'tfidf_panns_03_umbral':
        try:
            data = pd.read_csv(full_datos_path+'/tfidf_panns_03_umbral.csv')
        except: 
            data = pd.read_csv(full_datos_path+'/panns_tfidf_03_umbral.csv')

    elif variable == 'tfidf_panns_redondeo_x10':
        try:
            data = pd.read_csv(full_datos_path+'/tfidf_panns_redondeo_x10.csv')
        except: 
            data = pd.read_csv(full_datos_path+'/panns_tfidf_redondeo_x10.csv')

    ####################### Node2Vec
    ## YAMNet
    elif variable == 'node2vec_yamnet':
        data = np.load(full_datos_path+'/yamnet_node2vec.npy')

    elif variable == 'node2vec_yamnet_02_umbral':
        data = np.load(full_datos_path+'/yamnet_node2vec_02_umbral.npy')

    elif variable == 'node2vec_yamnet_03_umbral':
        data = np.load(full_datos_path+'/yamnet_node2vec_03_umbral.npy')

    elif variable == 'node2vec_yamnet_redondeo_x10':
        data = np.load(full_datos_path+'/yamnet_node2vec_redondeo_x10.npy')

    ## PANNs
    elif variable == 'node2vec_panns':
        data =  np.load(full_datos_path+'/panns_node2vec.npy')

    elif variable == 'node2vec_panns_02_umbral':
        data =  np.load(full_datos_path+'/panns_node2vec_02_umbral.npy')

    elif variable == 'node2vec_panns_03_umbral':
        data =  np.load(full_datos_path+'/panns_node2vec_03_umbral.npy')

    elif variable == 'node2vec_panns_redondeo_x10':
        data =  np.load(full_datos_path+'/panns_node2vec_redondeo_x10.npy')

    ################## Audio Tagging
    elif variable == 'yamnet_audio_tagging':
        data = pd.read_csv(full_datos_path+'/yamnet_audio_taggings.csv')
        data = data.reset_index(drop=True)
        data = data.iloc[:,:-1]
    
    elif variable == 'panns_audio_tagging':
        data = pd.read_csv(full_datos_path+'/panns_audio_tagging.csv')
        data = data.reset_index(drop=True)
        data = data.iloc[:,:-1]


    ################## LABELS
    if variable != 'labels' and variable[:8]!='node2vec' and variable!='yamnet_audio_tagging' and variable!='panns_audio_tagging':
        data = data[data['audio_name_chunk'].isin(chunks_names)]
        data = data.reset_index(drop=True)
        data = data.select_dtypes('float')

        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        data = np.nan_to_num(data)
        data = normalize(data)
    
    if variable != 'labels' and variable[:8]=='node2vec':

        data = process_node2vec(data, full_datos_path, variable, chunks_names)

        data = data.reset_index(drop=True)
        data = data.select_dtypes('float')

        data = np.nan_to_num(data)
        data = normalize(data)
    
    ################## Hacer PCA
    if variable == 'tfidf_yamnet' or variable == 'tfidf_panns' or variable == 'node2vec_yamnet' or variable == 'node2vec_panns'  or variable == 'tfidf_yamnet_03_umbral' or variable == 'tfidf_panns_03_umbral' or variable == 'node2vec_yamnet_03_umbral' or variable == 'node2vec_panns_03_umbral' or variable == 'tfidf_yamnet_02_umbral' or variable == 'tfidf_panns_02_umbral' or variable == 'node2vec_yamnet_02_umbral' or variable == 'node2vec_panns_02_umbral' or variable == 'tfidf_yamnet_redondeo_x10' or variable == 'node2vec_yamnet_redondeo_x10' or variable == 'tfidf_panns_redondeo_x10' or variable == 'node2vec_panns_redondeo_x10':

        pca = PCA(n_components=num_components)
        data = pca.fit_transform(data)
    
    return data

def process_node2vec(data, path_all_features, feat, 
                    chunks):

     ## TF-IDF
    if len(feat) > 15:

        try: # YAMNet
            dt_tfidf = pd.read_csv(f"{path_all_features}/{feat[9:9+6]}_tfidf_{feat[16:]}.csv").reset_index(drop=True)
        except:
            dt_tfidf = pd.read_csv(f"{path_all_features}/{feat[9:9+5]}_tfidf_{feat[15:]}.csv").reset_index(drop=True)

    else:
        dt_tfidf = pd.read_csv(f"{path_all_features}/{feat[9:]}_tfidf.csv").reset_index(drop=True)

    ## Node2Vec
    indices = dt_tfidf.index[dt_tfidf['audio_name_chunk'].isin(chunks)].tolist()
    node2vec = data[indices]

    node2vec = pd.DataFrame(node2vec.reshape(node2vec.shape[0],node2vec.shape[1]*node2vec.shape[2]))

    return node2vec

###########################################
##  SAVE CM
###########################################

def save_cm(cm, etiquetas_clases, path):

    # pal = sns.light_palette("#660033", reverse=True, as_cmap=True)

    font = {'fontname': 'serif'}

    cm = np.array(cm)
    cm = cm.astype(int)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True,fmt='d',
                xticklabels=etiquetas_clases, yticklabels=etiquetas_clases, cmap='Blues')
    plt.xticks(fontsize=12, **font)
    plt.yticks(fontsize=12, **font, rotation = 0)
    plt.savefig(path)


def save_curves(model,path):

    plt.figure(figsize=(18, 4))

    font = {'fontname': 'serif'}

    # Gráfico de pérdida de entrenamiento y validación
    plt.subplot(2, 2, 1)
    plt.plot(range(1, model.num_epochs + 1), model.train_loss, label='Train Loss')
    plt.plot(range(1, model.num_epochs + 1), model.valid_loss, label='Valid Loss')
    plt.xlabel('Epoch', **font)
    plt.ylabel('Loss', **font)
    plt.legend()
    plt.grid()
    plt.title('Training and Validation Loss', **font)

    # Gráfico de precisión
    plt.subplot(2, 2, 2)
    plt.plot(range(1, model.num_epochs + 1), model.train_acc, label='Train Accuracy')
    plt.plot(range(1, model.num_epochs + 1),  model.valid_acc, label='Valid Accuracy')
    plt.xlabel('Epoch', **font)
    plt.ylabel('Accuracy', **font)
    plt.legend()
    plt.title('Accuracy', **font)
    plt.grid()

    # Gráfico de F1 weighted
    plt.subplot(2, 2, 3)
    plt.plot(range(1, model.num_epochs + 1), model.f1_train_weighted, label='Train F1 weighted')
    plt.plot(range(1, model.num_epochs + 1),  model.f1_val_weighted, label='Valid F1 weighted')
    plt.xlabel('Epoch', **font)
    plt.ylabel('F1-score weighted', **font)
    plt.legend()
    plt.title('F1-score weighted', **font)
    plt.grid()
    plt.tight_layout()

    # Gráfico de F1 macro
    plt.subplot(2, 2, 4)
    plt.plot(range(1, model.num_epochs + 1), model.f1_train_macro, label='Train F1 macro')
    plt.plot(range(1, model.num_epochs + 1),  model.f1_val_macro, label='Valid F1 macro')
    plt.xlabel('Epoch', **font)
    plt.ylabel('F1-score macro', **font)
    plt.legend()
    plt.title('F1-score macro', **font)
    plt.grid()

    plt.tight_layout()
    plt.savefig(path)
