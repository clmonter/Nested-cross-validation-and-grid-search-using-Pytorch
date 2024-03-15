#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:44:10 2024

@author: clmonter
"""

from sklearn.model_selection import StratifiedKFold
from dataset_class import *
import pandas as pd
import torch
import MLP
import functions
import time

## Grid Search using own partitions
class GridSearch:

    def __init__(self, datos, labels,
                export_path,
                net_type,
                seed, cuda,
                scoring,
                p_grid, n_inner_folds=4,
                add_noise=True, 
                verbose_inner=False, plot_inner=False, export_all_process = True
                ):
        
        ## Datos y labels
        self.datos = datos
        self.labels = labels

        ## Export path
        self.export_path = export_path

        ## Network type
        self.net_type = net_type
        
        ## GPU
        self.seed = seed
        self.cuda = cuda

        ## Hyperparametros para cross validar
        self.p_grid = p_grid
        self.batch_size = p_grid['batch_size']
        self.hidden1 = p_grid['hidden1']
        self.num_epochs = p_grid['num_epochs']
        self.lr = p_grid['lr']
        self.l2 = p_grid['l2']
        self.loss_type = p_grid['loss_type']
        # self.network_type = p_grid['network_type']

        self.verbose_inner = verbose_inner
        self.plot_inner = plot_inner
        self.export_all_process = export_all_process

        self.n_inner_folds = n_inner_folds

        ## Añadir ruido gaussiano a los datos de entrenamiento
        self.add_noise = add_noise

        ## Parametros del propio Grid Search
        self.scoring = scoring

        ## Todos los modelos del GridSearch
        self.cv_models = {}

        ## Diccionario de DataFrames para comparar las inner folds entre sí
        self.dt_total = {}
        for i in range(1,self.n_inner_folds+1):
            key = f'inner_fold{i}'
            self.dt_total[key] = pd.DataFrame()

    def fit(self, outer_fold, variable):

        self.dimy = 8

        count_inner_fold = 1

        ###################################################
        ## Bucle para recorrer las INNER FOLDS
        ###################################################
        for n_inner in range(1, self.n_inner_folds+2):

            if n_inner == outer_fold:
                continue

            n_elementos = 0

            ## DataFrame final para hacer las comparaciones
            dt_inner_fold = pd.DataFrame(columns=list(self.p_grid.keys()) + [self.scoring])

            ## Train
            otros_valores = [x for x in range(1, self.n_inner_folds+2) if (x != n_inner and x!=outer_fold)]

            x_train_inner_list = []
            y_train_inner_list = []

            for train_inners in otros_valores:
                x_train_inner_aux = self.datos[f"fold_{train_inners}"]
                y_train_inner_aux = self.labels[f"fold_{train_inners}"]

                x_train_inner_list.append(x_train_inner_aux)
                y_train_inner_list.append(y_train_inner_aux)

            x_train_inner = np.concatenate(x_train_inner_list, axis=0)
            y_train_inner = np.concatenate(y_train_inner_list, axis=0)

            ## Validation
            x_val_inner = self.datos[f"fold_{n_inner}"]
            y_val_inner = self.labels[f"fold_{n_inner}"]

            self.dimx = x_train_inner.shape[1]

            ## Añadir ruido Gaussiano a los datos de entrenamiento de forma aleatoria
            if self.add_noise == True:
                noise_level = 0.1  # Define la desviación estándar del ruido gaussiano
                noise_ratio = 0.8  # Define la proporción de datos a los que se les añadirá ruido
                num_data_with_noise = int(len(x_train_inner) * noise_ratio)
                indices_to_add_noise = np.random.choice(len(x_train_inner), size=num_data_with_noise, replace=False)
                
                # Genera el ruido gaussiano
                gaussian_noise = np.random.normal(0, noise_level, x_train_inner.shape)
                
                # Añade el ruido gaussiano solo a los datos seleccionados
                #x_train_inner_with_noise = x_train_inner.copy()  # Copia los datos originales
                x_train_inner[indices_to_add_noise] += gaussian_noise[indices_to_add_noise]
                #x_train_inner = x_train_inner_with_noise

            '''
            if self.add_noise == True:
                gaussian_noise = np.random.normal(0, 1, x_train_inner.shape)
                x_train_inner = x_train_inner + gaussian_noise
            '''

            trainset = CreateDataset(y=np.array(y_train_inner), x=np.array(x_train_inner))
            validset = CreateDataset(y=np.array(y_val_inner), x=np.array(x_val_inner))

            ## 1. Batch size
            for batch in self.batch_size:

                trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
                validloader = torch.utils.data.DataLoader(validset, batch_size=batch, shuffle=True)

                for hidden in self.hidden1: ## 2. Hidden layers
                    for n_epoch in self.num_epochs: ## 3. Number of epochs
                        for learning_rate in self.lr: ## 3. Initial learning rate
                            for loss_t in self.loss_type: ## 4. Loss type
                                for l2_reg in self.l2: ## 5. L2 Ridge regularization

                                    model = MLP.MLP(net_type=self.net_type,
                                                    loss_type=loss_t, dimx=self.dimx, dimy=self.dimy, hidden1=hidden, num_epochs=n_epoch, lr=learning_rate, l2 = l2_reg,
                                                    labels=torch.tensor(y_train_inner),
                                                    verbose = self.verbose_inner, plot=self.plot_inner,
                                                    seed = self.seed, cuda = self.cuda)
                                    
                                    start_time = time.time()
                                    model.train_one_fold(trainloader,validloader)
                                    execution_time = time.time() - start_time

                                    self.cv_models[f'{n_elementos}'] = model

                                    ## Accuracy
                                    acc = model.valid_acc

                                    ## F1-weighted
                                    f1_weighted = model.f1_val_weighted

                                    ## F1-macro
                                    f1_macro = model.f1_val_macro

                                    if self.scoring == 'accuracy':
                                        valor_maximo = max(acc)
                                        posicion_maximo = acc.index(valor_maximo)
                                    
                                    elif self.scoring == 'f1-weighted':
                                        valor_maximo = max(f1_weighted)
                                        posicion_maximo = f1_weighted.index(valor_maximo)

                                    elif self.scoring == 'f1-macro':
                                        valor_maximo = max(f1_macro)
                                        posicion_maximo = f1_macro.index(valor_maximo)
                                        
                                    epoch_max = posicion_maximo + 1

                                    dt_inner_fold.loc[n_elementos,'inner_fold_id'] = int(n_inner)
                                    dt_inner_fold.loc[n_elementos,'batch_size'] = int(batch)
                                    dt_inner_fold.loc[n_elementos,'hidden1'] = int(hidden)
                                    dt_inner_fold.loc[n_elementos,'num_epochs'] = int(epoch_max)
                                    dt_inner_fold.loc[n_elementos,'lr'] = float(learning_rate)
                                    dt_inner_fold.loc[n_elementos,'l2'] = float(l2_reg)
                                    dt_inner_fold.loc[n_elementos,'loss_type'] = str(loss_t)
                                    dt_inner_fold.loc[n_elementos,'exection_time'] = float(execution_time)

                                    dt_inner_fold.loc[n_elementos,'accuracy'] = float(acc[posicion_maximo])
                                    dt_inner_fold.loc[n_elementos,'f1-weighted'] = float(f1_weighted[posicion_maximo])
                                    dt_inner_fold.loc[n_elementos,'f1-macro'] = float(f1_macro[posicion_maximo])

                                    dt_inner_fold.loc[n_elementos,f'{self.scoring}'] = float(valor_maximo)

                                    n_elementos = n_elementos + 1

            self.dt_total[f'inner_fold{count_inner_fold}'] = dt_inner_fold

            count_inner_fold = count_inner_fold + 1

        dataframes_list = [self.dt_total[f'inner_fold{i+1}'][f'{self.scoring}'] for i in range(self.n_inner_folds)]
        concatenated_df = pd.concat(dataframes_list, axis=1)

        if self.export_all_process == True:
            all_dataframes_list = [self.dt_total[f'inner_fold{i+1}'] for i in range(self.n_inner_folds)]
            all_concatenated_df = pd.concat(all_dataframes_list, axis=0)
            all_concatenated_df.to_csv(f"{self.export_path}/fold_{outer_fold}/{variable}_all_process.csv",index=False)

        media = concatenated_df.mean(axis=1)
        std = concatenated_df.std(axis=1)
        resultados = pd.DataFrame({'media': media, 'std': std})
        indice_max_media = resultados['media'].idxmax()

        self.best_params_ = self.dt_total[f'inner_fold1'].loc[indice_max_media,list(self.p_grid.keys())]
        self.best_model_ = self.cv_models[f'{indice_max_media}']

    def predict(self,x_test,y_test):

        testset = CreateDataset(y=np.array(y_test), x=np.array(x_test))
        testloader = torch.utils.data.DataLoader(testset, batch_size=int(self.best_params_['batch_size']), shuffle=True)
        metrics, self.fold_results_test = self.best_model_.predict(testloader)
        
        return metrics, self.fold_results_test

    def plot_cm(self,etiquetas_clases):

        self.best_model_.plot_cm(self.fold_results_test, etiquetas_clases)

    def plot_learning_curves(self):

        self.best_model_.plot_curves()

## Clasical Grid Search
'''
class GridSearch:

    def __init__(self, net_type,
                 seed, cuda,
                 p_grid, n_inner_folds=2, scoring='f1-weighted',
                 add_noise=True, 
                 verbose_inner=False, plot_inner=False,
                 ):
        
        ## Network type
        self.net_type = net_type
        
        ## GPU
        self.seed = seed
        self.cuda = cuda

        ## Hyperparametros para cross validar
        self.p_grid = p_grid
        self.batch_size = p_grid['batch_size']
        self.hidden1 = p_grid['hidden1']
        self.num_epochs = p_grid['num_epochs']
        self.lr = p_grid['lr']
        self.loss_type = p_grid['loss_type']
        # self.network_type = p_grid['network_type']

        self.verbose_inner = verbose_inner
        self.plot_inner = plot_inner

        self.n_inner_folds = n_inner_folds

        ## Añadir ruido gaussiano a los datos de entrenamiento
        self.add_noise = add_noise

        ## Parametros del propio Grid Search
        self.cv_inner = StratifiedKFold(n_splits=self.n_inner_folds, shuffle=True, random_state=42)    
        self.scoring = scoring

        ## Todos los modelos del GridSearch
        self.cv_models = {}

        ## Diccionario de DataFrames para comparar las inner folds entre sí
        self.dt_total = {}
        for i in range(1,self.n_inner_folds+1):
            key = f'inner_fold{i}'
            self.dt_total[key] = pd.DataFrame()

    def fit(self,x_train,y_train):

        self.dimx = x_train.shape[1]
        self.dimy = 10

        count_inner_fold = 1

        ###################################################
        ## Bucle para recorrer las INNER FOLDS
        ###################################################
        for train_index, val_index in self.cv_inner.split(x_train, y_train): 

            n_elementos = 0

            ## DataFrame final para hacer las comparaciones
            dt_inner_fold = pd.DataFrame(columns=list(self.p_grid.keys()) + [self.scoring])

            ## Preparar dataloaders
            x_train_inner, x_val_inner = x_train[train_index,:], x_train[val_index,:]
            y_train_inner, y_val_inner = y_train[train_index], y_train[val_index]

            ## Añadir ruido Gaussiano a los datos de entrenamiento
            if self.add_noise == True:
                gaussian_noise = np.random.normal(0, 1, x_train_inner.shape)
                x_train_inner = x_train_inner + gaussian_noise

            trainset = CreateDataset(y=np.array(y_train_inner), x=np.array(x_train_inner))
            validset = CreateDataset(y=np.array(y_val_inner), x=np.array(x_val_inner))

            ## 1. Batch size
            for batch in self.batch_size:

                trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=True)
                validloader = torch.utils.data.DataLoader(validset, batch_size=batch, shuffle=True)

                for hidden in self.hidden1: ## 2. Hidden layers
                    for n_epoch in self.num_epochs: ## 3. Number of epochs
                        for learning_rate in self.lr: ## 3. Initial learning rate
                            for loss_t in self.loss_type: ## 4. Loss type

                                model = MLP.MLP(net_type=self.net_type,
                                                loss_type=loss_t, dimx=self.dimx, dimy=self.dimy, hidden1=hidden, num_epochs=n_epoch, lr=learning_rate, labels=torch.tensor(y_train),
                                                verbose = self.verbose_inner, plot=self.plot_inner,
                                                seed = self.seed, cuda = self.cuda)

                                model.train_one_fold(trainloader,validloader)

                                self.cv_models[f'{n_elementos}'] = model

                                if self.scoring == 'accuracy':
                                    acc = model.valid_acc
                                    valor_maximo = max(acc)
                                    posicion_maximo = acc.index(valor_maximo)
                                
                                elif self.scoring == 'f1-weighted':
                                    f1 = model.f1_val_weighted
                                    valor_maximo = max(f1)
                                    posicion_maximo = f1.index(valor_maximo)

                                elif self.scoring == 'f1-macro':
                                    f1 = model.f1_val_macro
                                    valor_maximo = max(f1)
                                    posicion_maximo = f1.index(valor_maximo)
                                    
                                epoch_max = posicion_maximo + 1

                                dt_inner_fold.loc[n_elementos,'batch_size'] = int(batch)
                                dt_inner_fold.loc[n_elementos,'hidden1'] = int(hidden)
                                dt_inner_fold.loc[n_elementos,'num_epochs'] = int(epoch_max)
                                dt_inner_fold.loc[n_elementos,'lr'] = float(learning_rate)
                                dt_inner_fold.loc[n_elementos,'loss_type'] = str(loss_t)
                                dt_inner_fold.loc[n_elementos,f'{self.scoring}'] = float(valor_maximo)

                                n_elementos = n_elementos + 1

            self.dt_total[f'inner_fold{count_inner_fold}'] = dt_inner_fold

            count_inner_fold = count_inner_fold + 1

        dataframes_list = [self.dt_total[f'inner_fold{i+1}'][f'{self.scoring}'] for i in range(self.n_inner_folds)]
        concatenated_df = pd.concat(dataframes_list, axis=1)
        media = concatenated_df.mean(axis=1)
        std = concatenated_df.std(axis=1)
        resultados = pd.DataFrame({'media': media, 'std': std})
        indice_max_media = resultados['media'].idxmax()

        self.best_params_ = self.dt_total[f'inner_fold1'].loc[indice_max_media,list(self.p_grid.keys())]
        self.best_model_ = self.cv_models[f'{indice_max_media}']

    def predict(self,x_test,y_test):

        testset = CreateDataset(y=np.array(y_test), x=np.array(x_test))
        testloader = torch.utils.data.DataLoader(testset, batch_size=int(self.best_params_['batch_size']), shuffle=True)
        metrics, self.fold_results_test = self.best_model_.predict(testloader)
        
        return metrics, self.fold_results_test

    def plot_cm(self,etiquetas_clases):

        self.best_model_.plot_cm(self.fold_results_test, etiquetas_clases)

    def plot_learning_curves(self):

        self.best_model_.plot_curves()

'''
