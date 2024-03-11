import argparse
import torch
import numpy as np
import random
import os
import functions
from grid_search import *
import pickle
from sklearn.metrics import confusion_matrix
import time

import sys
sys.path.append('/home/cmramirez/Desktop/Python/PAPER_dic_2023/9_1_4INNER_PYTORCH_NESTED_CROSS_VAL/interspeech_utils') 
from confidence_i import evaluate_with_conf_int

########################################################
## Argument parser
########################################################

parser = argparse.ArgumentParser()

######## Features
parser.add_argument('--feature', type=str, default='panns_audio_tagging',
                    choices=['librosa','timbral',
                             'hog','lbp', ## Spectrogram features
                             'tfidf_yamnet','tfidf_panns','node2vec_yamnet','node2vec_panns',
                             'tfidf_yamnet_03_umbral', 'node2vec_yamnet_03_umbral', # 0.3 estático
                             'tfidf_panns_03_umbral', 'node2vec_panns_03_umbral',
                             'tfidf_yamnet_02_umbral', 'node2vec_yamnet_02_umbral', # 0.2 estático
                             'tfidf_panns_02_umbral', 'node2vec_panns_02_umbral',
                             'tfidf_yamnet_redondeo_x10', 'node2vec_yamnet_redondeo_x10', # conteo x 10
                             'tfidf_panns_redondeo_x10', 'node2vec_panns_redondeo_x10',
                             'yamnet_audio_tagging','panns_audio_tagging'], ## Audio Tagging
                    help='Feature group con el que vamos a hacer la nested cross validation.')

######## GridSearch
parser.add_argument('--net_type', type=str, default= 'SimpleNet',
                    choices = ['SimpleNet', 'MultiHeadNet'],
                    help='Tipo de red que vamos a utilizar')

parser.add_argument('--scoring', default='f1-macro',
                    choices=['accuracy','f1-macro','f1-weighted'],
                    help='Metrica que vamos a utilizar para escoger los mejores parametros')

parser.add_argument('--p_grid', type=dict,
                    default = {
                        'batch_size':[24],
                        'hidden1':[32,64,128],
                        'num_epochs':[40],
                        'lr':[0.001],
                        'l2':[0.01],
                        'loss_type':['weighted', 'non_weighted', 'sqrt_weighted', 'log_weighted']
                    })

######## Preprocessing
parser.add_argument('--pca_n_components', type=int, default=40,
                    help = 'Number of components of the PCA, por Acoustic Unit Descriptors.')
parser.add_argument('--add_noise', type=bool, default=True,
                    help='Si queremos añadir ruido Gaussiano a los datos de entrenamiento, media=0 y std=1.')

########  Save results
parser.add_argument('--save_cm_fold', type=bool, default=False,
                    help='Si queremos guardar las matrices de confusion para cada fold.')
parser.add_argument('--save_cm', type=bool, default=True,
                    help='Si queremos guardar las matrices de confusion acumuladas.')

parser.add_argument('--save_hyperparams', type=bool, default=True,
                    help='Si queremos guardar los resultados y los mejores hiperparametros.')

parser.add_argument('--save_learning_curves_fold', type=bool, default=True,
                    help='Si queremos guardar las curvas de aprendizaje para cada fold.')

parser.add_argument('--save_predictions', type=bool, default=True,
                    help='Si queremos guardar un diccionario de todos los resultados reales y predichos.')

parser.add_argument('--save_models', type=bool, default=True,
                    help='Si queremos guardar los mejores modelos para cada fold.')

########  Outer and inner folds 
parser.add_argument('--n_folds', type=int, default = 5,
                    help='Number of folds in the outer loop.')
parser.add_argument('--inner_folds', type=int, default = 4,
                    help='Number of folds in the inner loop.')

########  GPU 
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
parser.add_argument('--gpuID', type=int, default=0,
                    help='set gpu id to use (default: 0)')
parser.add_argument('--seed', type=int, default=42, 
                    help='random seed (0). Manual seed (42). Si queremos que sea reproducible hay que usar una semilla manual.')

######## Verbose and plots 
#### Inner CV
## Verbose
parser.add_argument('--verbose_inner_cv', type=bool, default = True,
                    help='Si queremos que vaya imprimiendo los resultados durante la validación cruzada')
## Print loss curves
parser.add_argument('--curves_inner_cv', type=bool, default = False,
                    help='Si queremos que vaya imprimiendo las curvas de entrenamiento cada vez que entrena un modelo')
## Exportar los resultados con distintos hiperparametros
parser.add_argument('--export_all_process', type=bool, default = True,
                    help='Si queremos exportar todos los resultados con distintos hiperparámetros')

#### Outer CV
## Verbose
parser.add_argument('--verbose_outer_cv', type=bool, default = True,
                    help='Si queremos que vaya imprimiendo los resultados para cada outer fold')
## Print loss curves
parser.add_argument('--curves_outer_cv', type=bool, default = False,
                    help='Si queremos que vaya imprimiendo las curvas de entrenamiento para cada outer fold')

########################################################
## Paths
########################################################
######## Paths to export results
## Local
parser.add_argument('--path_cm', type=str, 
                   default='your_path_to_export_confussion_matrices',
                   help='Path to export confussion matrices (.png))
parser.add_argument('--path_hyperparams_results', type=str, 
                   default='your_path_to_export_hyperparameters',
                   help='Path to export hyperparameter results for each fold (.txt)')
parser.add_argument('--path_learning_curves',type=str, 
                   default='your_path_to_export_learning_curves',
                   help='Path to export hyperparameter learning curves for each fold (.png)')
parser.add_argument('--export_path', type=str, 
                    default='your_path_to_export_all_inner_fold_scores',
                    help='Path to export a table for all inner fold scores with each set of hyperparameters (.csv)')
parser.add_argument('--path_predictons', type=str, 
                    default='your_path_to_export_test_predictions',
                    help='Path to export test predictions for each outer fold.')
parser.add_argument('--path_models', type=str,
                    default='your_path_to_save_models',
                    help='Path to save the best model for each outer fold.')
######## Paths to data and labels
## Local
parser.add_argument('--full_data_path', type=str, 
                    default='path_to_your_data')
parser.add_argument('--labels_path', type=str, 
                    default='path_to_your_labels')

args = parser.parse_args()

if args.cuda == 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpuID)

# Seed
SEED = args.seed
if SEED == 0: # Vamos a usar el 0 como random seed
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
if args.cuda:
    torch.cuda.manual_seed(SEED)
else:
    torch.manual_seed(SEED)

########################################################
## Parametros estaticos
########################################################

etiquetas_clases = {'Home': 0, 'Caf': 1, 'Resto': 2, 'Transp': 3, 'Bar': 4, 'Work': 5, 'MedC': 6, 'School': 7}

########################################################
## Cargar datos
########################################################

datos = {}
labels = {}

## Cargar todos los datos y guardarlos en diccionarios
if args.verbose_outer_cv == True:
    print('Preparing data using k fold partition...')

for fold in range(1,args.n_folds+1):
    datos[f'fold_{fold}'] = functions.load_variable(args.full_data_path, args.labels_path, fold, args.feature, num_components=args.pca_n_components)
    labels[f'fold_{fold}'] = functions.load_variable(args.full_data_path, args.labels_path, fold, variable='labels', num_components=args.pca_n_components)

########################################################
## Outer loop
########################################################

if args.save_cm == True:
    cm_acum = np.zeros((8,8))

for fold in range(1,args.n_folds+1):

    if args.verbose_outer_cv == True:
        print(f'Starting cross validating fold {fold}')

    grid = GridSearch(datos,labels,
                      args.export_path,
                      args.net_type,
                      args.seed, args.cuda,
                      args.scoring,
                      args.p_grid, args.inner_folds, add_noise=True,
                      verbose_inner=args.verbose_inner_cv, plot_inner=args.curves_inner_cv, export_all_process=args.export_all_process)

    x_test = datos[f'fold_{fold}']
    y_test = labels[f'fold_{fold}']

    grid.fit(fold, args.feature)

    start_time = time.time()
    metrics, results = grid.predict(x_test,y_test)
    inference_time = time.time() - start_time

    predicted_labels = np.array(results['predichos'])
    real_labels = np.array(results['reales'])
    conf_acc = evaluate_with_conf_int(predicted_labels,  metric='accuracy_score', labels=real_labels)
    conf_f1_macro = evaluate_with_conf_int(predicted_labels, metric='f1-score-macro', labels=real_labels)
    conf_f1_weighted = evaluate_with_conf_int(predicted_labels, metric='f1-score-weighted', labels=real_labels)

    ## Opciones de visualizacion
    if args.save_cm == True:
        cm_fold = confusion_matrix(results['reales'], results['predichos'])
        cm_acum = cm_acum + cm_fold

    if args.save_cm_fold == True:
        functions.save_cm(results, etiquetas_clases, f"{args.path_cm}/{args.feature}_fold_{fold}_sgd.png")

    if args.verbose_outer_cv == True:
        print(f'For fold {fold} on feature {args.feature}, best params are: {grid.best_params_}\n{metrics}')

    if args.save_hyperparams == True:
        with open(f"{args.path_hyperparams_results}/{args.feature}_sgd.txt", 'a') as archivo:
            archivo.write(f'\nFor fold {fold} on feature {args.feature}, best params are: {grid.best_params_}\n{metrics}\n')
            archivo.write(f'\nConfidance intervals: \nAccuracy: {conf_acc}\nF1-weighted: {conf_f1_weighted}\nF1-macro: {conf_f1_macro}\n')
            archivo.write(f'\nNumber of parameters in the model: {sum(p.numel() for p in grid.best_model_.parameters())}\n')
            archivo.write(f'\nInference time: {inference_time} (s)\n')
            archivo.write(f'\n---------------------------\n')

    if args.save_predictions == True:
        with open(f'{args.path_predictons}/{args.feature}_fold_{fold}_sgd.pkl', 'wb') as archivo:
            pickle.dump(results, archivo)
    
    if args.save_models == True:
        torch.save(grid.best_model_.state_dict(), f'{args.path_models}/{args.feature}_fold_{fold}_sgd.pth')

    if args.curves_outer_cv == True:
        grid.plot_cm(etiquetas_clases)
        grid.plot_learning_curves()

    if args.save_learning_curves_fold == True:
        functions.save_curves(grid.best_model_, f"{args.path_learning_curves}/{args.feature}_fold_{fold}_sgd.png")

if args.save_cm == True:
    functions.save_cm(cm_acum, etiquetas_clases, f"{args.path_cm}/{args.feature}_total_sgd.png")





