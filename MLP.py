#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 09:44:10 2024

@author: clmonter
"""

import torch
from torch import nn
import network as net
import functions
import random
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
from IPython.display import clear_output
import seaborn as sns

class MLP(nn.Module):
    def __init__(self, 
                 net_type,
                 loss_type, dimx, dimy, hidden1, num_epochs, lr, l2, labels, 
                 verbose, plot,
                 seed, cuda):
        
        super(MLP, self).__init__()
        
        ## Tipo de red
        self.net_type = net_type

        ## Estructura
        self.dimx = dimx
        self.dimy = dimy
        self.hidden1 = hidden1

        ## Other hyperparameters
        self.num_epochs = num_epochs
        self.lr = lr
        self.l2 = l2

        ## GPU y semilla
        self.seed = seed
        self.cuda = cuda

        ## Visualización
        self.verbose = verbose
        self.plot = plot

        if loss_type == 'weighted':

            class_samples = torch.bincount(labels)
            total_samples = float(labels.size(0))
            class_weights = total_samples / (class_samples) #+ 1e-6)  # Se agrega 1e-6 para evitar divisiones por cero
            class_weights = class_weights / class_weights.sum()

            if self.cuda:
                class_weights = class_weights.cuda()

            self.loss_function = nn.CrossEntropyLoss(weight=class_weights)

        elif loss_type == 'log_weighted':

            class_samples = torch.bincount(labels)
            total_samples = float(labels.size(0))
            class_weights = total_samples / (class_samples) #+ 1e-6)  # Se agrega 1e-6 para evitar divisiones por cero
            log_class_weights = torch.log(class_weights + 1) 
            log_class_weights = log_class_weights / log_class_weights.sum()

            if self.cuda:
                log_class_weights = log_class_weights.cuda()

            self.loss_function = nn.CrossEntropyLoss(weight=log_class_weights)

        elif loss_type == 'sqrt_weighted':

            class_samples = torch.bincount(labels)
            total_samples = float(labels.size(0))
            class_weights = total_samples / torch.sqrt(class_samples.float())  # Aplicando la raíz cuadrada inversa
            class_weights = class_weights / class_weights.sum()

            if self.cuda:
                class_weights = class_weights.cuda()

            self.loss_function = nn.CrossEntropyLoss(weight=class_weights)

        elif loss_type == 'non_weighted':

            self.loss_function = nn.CrossEntropyLoss()

        ## Loss
        self.train_loss = []
        self.valid_loss = []

        ## Accuracy
        self.train_acc = []
        self.valid_acc = []
        self.test_acc = []

        ## F1-score weighted
        self.f1_train_weighted = []
        self.f1_val_weighted = []
        self.f1_test_weighted = []

        ## F1-score macro
        self.f1_train_macro = []
        self.f1_val_macro = []
        self.f1_test_macro = []

        ## Todos los resultados
        self.fold_results_train = {'predichos': [], 'reales': []}
        self.fold_results_val = {'predichos': [], 'reales': []}
        self.fold_results_test = {'predichos': [], 'reales': []}

    def train_one_fold(self, trainloader, validloader):
        
        # Seed
        SEED = self.seed
        if SEED == 0: # Vamos a usar el 0 como random seed
            np.random.seed(SEED)
            random.seed(SEED)
            torch.manual_seed(SEED)
        if self.cuda:
            torch.cuda.manual_seed(SEED)
        else:
            torch.manual_seed(SEED)

        if self.net_type == 'SimpleNet':
            self.network = net.SimpleNet(dimx=self.dimx, hidden1=self.hidden1 ,dimy=self.dimy)
        elif self.net_type == 'MultiHeadNet':
            self.network = net.MultiHeadNet(dimx=self.dimx, hidden1=self.hidden1 ,dimy=self.dimy)

        if self.cuda:
            self.network = self.network.cuda()

        # network.apply(functions.reset_weights)

        # Initialize optimizer
        # optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr, weight_decay=self.l2)

        optimizer = torch.optim.SGD(self.network.parameters(), momentum=0.9, lr=0.001, weight_decay=0.0001, nesterov=True)
        
        #############################################
        ## Training during epochs
        #############################################

        for epoch in range(0, self.num_epochs):

            # Para otras metricas
            all_targets = []
            all_predicted = []

            val_all_targets = []
            val_all_predicted = []
            
            # Set current loss value
            current_loss = 0.0
            total = 0
            correct = 0

            valid_running_loss = 0.0
            total_valid = 0
            correct_valid = 0

        #############################################
        ## Iterate over batches
        #############################################

            for i, batch in enumerate(trainloader, 0):

                if self.cuda == 1:
                    inputs = batch['data'].cuda()
                    targets = batch['label'].cuda()
                else:
                    inputs = batch['data']
                    targets = batch['label']

                # Zero the gradients
                optimizer.zero_grad()
                
                # Perform forward pass
                outputs = self.network(inputs)
                _, predicted = torch.max(outputs.data, 1)

                ## Fold results
                if epoch == self.num_epochs-1:
                    self.fold_results_train['predichos'].extend(predicted.cpu().tolist())
                    self.fold_results_train['reales'].extend(targets.cpu().tolist())
                
                # Compute loss
                loss = self.loss_function(outputs, targets)
                
                # Perform backward pass
                loss.backward()
                
                # Perform optimization
                optimizer.step()
                
                # Print statistics
                current_loss += loss.item()

                ## Accuracy
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                ## Para otras metricas usando sklearn
                all_targets.extend(targets.cpu().numpy())
                all_predicted.extend(predicted.cpu().numpy())

            self.train_loss.append(current_loss)
            self.train_acc.append(100.0 * correct / total)

            self.f1_train_weighted.append(f1_score(all_targets, all_predicted, average='weighted'))
            self.f1_train_macro.append(f1_score(all_targets, all_predicted, average='macro'))

            ############################### 
            ## VALIDATION
            with torch.no_grad():
                for i, batch in enumerate(validloader, 0):

                    if self.cuda == 1:
                        inputs = batch['data'].cuda()
                        targets = batch['label'].cuda()
                    else:
                        inputs = batch['data']
                        targets = batch['label']

                    outputs = self.network(inputs)
                    _, predicted = torch.max(outputs.data, 1)

                    ## Fold results
                    if epoch == self.num_epochs-1:
                        self.fold_results_val['predichos'].extend(predicted.cpu().tolist())
                        self.fold_results_val['reales'].extend(targets.cpu().tolist())

                    valid_loss_aux = self.loss_function(outputs, targets)
                    valid_running_loss += valid_loss_aux.item()

                    ## Accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_valid += targets.size(0)
                    correct_valid += (predicted == targets).sum().item()

                    val_all_targets.extend(targets.cpu().numpy())
                    val_all_predicted.extend(predicted.cpu().numpy())

                self.valid_loss.append(valid_running_loss)
                self.valid_acc.append(100.0 * correct_valid / total_valid)

                self.f1_val_weighted.append(f1_score(val_all_targets, val_all_predicted, average='weighted'))
                self.f1_val_macro.append(f1_score(val_all_targets, val_all_predicted, average='macro'))


            # Print epoch train and validation loss
            if self.verbose == True:
                print(f"Epoch {epoch+1}:\nTraining Loss {current_loss}, Acc {100.0 * correct / total}, F1-weighted {f1_score(all_targets, all_predicted, average='weighted')}, F1-macro {f1_score(all_targets, all_predicted, average='macro')}")
                print(f"Validation Loss {valid_running_loss}, Acc {100.0 * correct_valid / total_valid}, F1-weighted {f1_score(val_all_targets, val_all_predicted, average='weighted')}, F1-macro {f1_score(val_all_targets, val_all_predicted, average='macro')}")
                print("\n")
                        
        # Process is complete.
        if self.verbose == True:
            print('Training process has finished.')

        if self.plot == True:
            self.plot_curves()

    def predict(self,testloader):

        correct, total = 0, 0
        test_all_targets = []
        test_all_predicted = []

        with torch.no_grad():

            for i, batch in enumerate(testloader, 0):

                if self.cuda == 1:
                    inputs = batch['data'].cuda()
                    targets = batch['label'].cuda()
                else:
                    inputs = batch['data']
                    targets = batch['label']

                outputs = self.network.forward(inputs)

                # Set total and correct
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                test_all_targets.extend(targets.cpu().numpy())
                test_all_predicted.extend(predicted.cpu().numpy())

                ## Fold results
                self.fold_results_test['predichos'].extend(predicted.cpu().tolist())
                self.fold_results_test['reales'].extend(targets.cpu().tolist())

            self.test_acc.append(100.0 * correct / total)

            self.f1_test_weighted.append(f1_score(test_all_targets, test_all_predicted, average='weighted'))
            self.f1_test_macro.append(f1_score(test_all_targets, test_all_predicted, average='macro'))

            metrics = {'accuracy':self.test_acc,
                       'f1_score_weighted':self.f1_test_weighted,
                       'f1_score_macro':self.f1_test_macro}
            
            return metrics, self.fold_results_test


    def plot_curves(self):

        plt.figure(figsize=(18, 4))

        font = {'fontname': 'serif'}

        # Gráfico de pérdida de entrenamiento y validación
        plt.subplot(2, 2, 1)
        plt.plot(range(1, self.num_epochs + 1), self.train_loss, label='Train Loss')
        plt.plot(range(1, self.num_epochs + 1), self.valid_loss, label='Valid Loss')
        plt.xlabel('Epoch', **font)
        plt.ylabel('Loss', **font)
        plt.legend()
        plt.grid()
        plt.title('Training and Validation Loss', **font)

        # Gráfico de precisión
        plt.subplot(2, 2, 2)
        plt.plot(range(1, self.num_epochs + 1), self.train_acc, label='Train Accuracy')
        plt.plot(range(1, self.num_epochs + 1),  self.valid_acc, label='Valid Accuracy')
        plt.xlabel('Epoch', **font)
        plt.ylabel('Accuracy', **font)
        plt.legend()
        plt.title('Accuracy', **font)
        plt.grid()

        # Gráfico de F1 weighted
        plt.subplot(2, 2, 3)
        plt.plot(range(1, self.num_epochs + 1), self.f1_train_weighted, label='Train F1 weighted')
        plt.plot(range(1, self.num_epochs + 1),  self.f1_val_weighted, label='Valid F1 weighted')
        plt.xlabel('Epoch', **font)
        plt.ylabel('F1-score weighted', **font)
        plt.legend()
        plt.title('F1-score weighted', **font)
        plt.grid()
        plt.tight_layout()

        # Gráfico de F1 macro
        plt.subplot(2, 2, 4)
        plt.plot(range(1, self.num_epochs + 1), self.f1_train_macro, label='Train F1 macro')
        plt.plot(range(1, self.num_epochs + 1),  self.f1_val_macro, label='Valid F1 macro')
        plt.xlabel('Epoch', **font)
        plt.ylabel('F1-score macro', **font)
        plt.legend()
        plt.title('F1-score macro', **font)
        plt.grid()

        plt.tight_layout()
        plt.show()


    def plot_cm(self, dict, etiquetas_clases):

        pal = sns.light_palette("#660033", reverse=True, as_cmap=True)

        font = {'fontname': 'serif'}

        cm = confusion_matrix(dict['reales'], dict['predichos'])

        plt.figure(figsize=(8, 5))
        sns.heatmap(cm, annot=True,xticklabels=etiquetas_clases, yticklabels=etiquetas_clases, cmap=pal, fmt='.2f')
        plt.xticks(fontsize=12, **font)
        plt.yticks(fontsize=12, **font, rotation = 0)
        plt.show()
        


