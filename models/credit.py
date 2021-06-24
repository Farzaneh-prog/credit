import os
import sys

import numpy as np
import pandas as pd
import pickle as pi
from sklearn.svm import SVC

#working directory: /home/farzaneh/DataScientist/Projekt/scheidung/models/Divorce.py

current_dir = 'Credit'
sys.path.append('../src/features/')
import build_features
from build_features import *
sys.path.append('../src/models/')
import train_model
from train_model import *
sys.path.append('../src/visualization/')
import visualize
from visualize import *
#https://stackabuse.com/creating-and-importing-modules-in-python/


#reading Data
def __readheader(filehandle, numberheaderlines=1):
    """Reads the specified number of lines and returns the comma-delimited 
    strings on each line as a list"""
    for _ in range(numberheaderlines):
        yield map(str.strip, filehandle.readline().strip().split(' '))

with open('../data/processed/SouthGermanCredit.asc', 'r') as rh:
    # Single header line
    A = next(__readheader(rh))
    head = (list(A))
    datanp = np.genfromtxt(rh, delimiter=' ')

data = pd.DataFrame(datanp, columns=head)

#preparing Data for modelling
preprocessor = Preprocessor(data)
'''
Akkuranz = []

for i in range (2,9):
    test_size = 0.1*i
    X_train, X_test, y_train, y_test, X, y = preprocessor.get_data(test_size)

    #Data Statistics
    count_positive, count_negative, percentage_positive, percentage_negative = preprocessor.statistica()

    #choosing the model
    algo="DT" # algo = ["DT","rf","knn","svm"]

    #training the model
    trained_model, model_confusion_mat = train_model(X_train, X_test, y_train, y_test, algo)

    #model optimisation
    if algo == "svm":
        best_model_parameters = SVC(C=0.1, gamma='auto', kernel='poly', probability=True)
    else:
        best_model_parameters = optimize_model(trained_model, X_train, y_train, algo)

    best_model, best_model_confusion_mat, mean_accuracy_best_parameters = kfoldevaluate_optimzed_model(algo, best_model_parameters, X, y, X_train, X_test, y_train, y_test)

    Akkuranz.append(round(mean_accuracy_best_parameters,4)*100)

#Akkuranz plot
Akkuranz_pl(Akkuranz, algo)
'''    
#Either the top one is commented or the one bellow, above is a search for the best train/test size and bellow is modelling with the best calculated variables, with all the different methods
i = 5
test_size = 0.1*i
X_train, X_test, y_train, y_test, X, y = preprocessor.get_data(test_size)

#Data Statistics
count_positive, count_negative, percentage_positive, percentage_negative = preprocessor.statistica()

print('Cases who did return their credit in percent: {}%'.format(round(percentage_positive,3)))     #class one
print('Cases who did not return their credit in percent: {}%'.format(round(percentage_negative,4)))     #class zero

#choosing the model
algorithms = ["DT","rf","knn","svm"]
trained_model_list = []
model_confusion_mat_list = []
best_model_parameters_list = []
best_model_list = []
best_model_confusion_mat_list = []
mean_accuracy_best_parameters_list = []
fpr_list = []
tpr_list = []
threshold_list = []
roc_auc_list = []

for i in range (0,3): # 4 is for svm
    algo = algorithms[i]

    #training the model
    trained_model, model_confusion_mat = train_model(X_train, X_test, y_train, y_test, algo)

    trained_model_list.append(trained_model)
    #model_confusion_mat_list.append(model_confusion_mat)

    #model optimisation
    if algo == "svm":
        best_model_parameters = SVC(C=1.0, gamma='auto', kernel='linear', probability=True)
    else:
        best_model_parameters = optimize_model(trained_model, X_train, y_train, algo)

    best_model_parameters_list.append(best_model_parameters)

    best_model, best_model_confusion_mat, mean_accuracy_best_parameters = kfoldevaluate_optimzed_model(algo, best_model_parameters, X, y, X_train, X_test, y_train, y_test)
    best_model_list.append(best_model)
    best_model_confusion_mat_list.append(best_model_confusion_mat)
    mean_accuracy_best_parameters_list.append(mean_accuracy_best_parameters)

    

    #ROC calculation
    fpr, tpr, threshold, roc_auc = roc_vorbereitung(best_model, X_test, y_test)
    fpr_list.append(fpr)
    tpr_list.append(tpr)
    threshold_list.append(threshold)
    roc_auc_list.append(roc_auc)

    #graphical output
    pl_ROC(algo, roc_auc, fpr, tpr)
    if algo=="DT": Diabetes_tree(best_model, X)

    #save in Pickle file
    file_name = "classification_model"+algo+".pickle"
    fill = open(file_name,'wb')     #allow to Write the file in a Binary format
    pi.dump(best_model,fill)
    fill.close()
    #comment it till here

vote_method = ['hard', 'soft','softWithWeight'] 
#method_Weight = [2,3,1,4] # for [DT,rf,knn,svm]
method_Weight = [2,4,1] # for [DT,rf,knn]

for i in range(0,3):
    vote = vote_method[i]
    Ensemblemodel, conf_mat = Ensembel(best_model_list, X, y, X_train, X_test, y_train, y_test, vote, method_Weight)
    best_model_list.append(Ensemblemodel)
    best_model_confusion_mat_list.append(conf_mat)

    #save in Pickle file
    file_name = "Ensemble_model"+vote+".pickle"
    fill = open(file_name,'wb')     #allow to Write the file in a Binary format
    pi.dump(Ensemblemodel,fill)
    fill.close()
    
    if i > 0 : 
        #ROC calculation
        fpr, tpr, threshold, roc_auc = roc_vorbereitung(Ensemblemodel, X_test, y_test)
        #graphical output
        pl_ROC(vote, roc_auc, fpr, tpr)

        fpr_list.append(fpr)
        tpr_list.append(tpr)
        threshold_list.append(threshold)
        roc_auc_list.append(roc_auc)


#!!! This part is really tricky
print(' tpr5: {}, \n fpr5: {}, \n threshold5: {}, \n tpr6: {}, \n fpr6: {}, \n threshold6: {}'.format(tpr_list[3],fpr_list[3],threshold_list[3],tpr_list[4],fpr_list[4],threshold_list[4]))

Cutoffthreshold = 0.54

for i in range(4,6):
    model_pred, conf_mat = predict_with_cutoff(best_model_list, Cutoffthreshold, X_test, y_test, i)
    best_model_confusion_mat_list.append(conf_mat)

ConfmatPl(best_model_confusion_mat_list, algorithms[0:4], vote_method)
