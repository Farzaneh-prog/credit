import os
import pandas as pd
import numpy as np
import pickle as pi

#working directory: src/models/predict_model.py

#save in Pickle file
path_start = os.getcwd()
pathr = os.path.dirname(os.getcwd())+'/../models'
os.chdir(pathr)

#choosing the model
algo="rf" # algo = ["DT","rf","knn","svm"]
file_name = "classification_model"+algo+".pickle"
#file_name = "Ensemble_modelsoftWithWeight.pickle"

fill = open(file_name,'rb')     #read only file in a Binary format
classifier = pi.load(fill)
fill.close()

#change to the start working directory
os.chdir(path_start)

pathread = os.path.dirname(os.path.dirname(os.getcwd()))+'/data/external/Predict.asc'

#reading Data
def __readheader(filehandle, numberheaderlines=1):
    """Reads the specified number of lines and returns the comma-delimited 
    strings on each line as a list"""
    for _ in range(numberheaderlines):
        yield map(str.strip, filehandle.readline().strip().split(' '))

with open(pathread, 'r') as rh:
    # Single header line
    A = next(__readheader(rh))
    head = (list(A))
    datanp = np.genfromtxt(rh, delimiter=' ')

data = pd.DataFrame(datanp, columns=head)

#Get predicted values from test data 
y_pred = classifier.predict(data) 
#threshold = 0.57      # When we want to cut the data at a specific threshold
#y_pred = (classifier.predict_proba(data)[:,1] >= threshold).astype(bool)
#print(y_pred)


erg = pd.DataFrame(y_pred, columns = ['prediction'])
ergebnis = pd.concat([data,erg],axis=1)
print(ergebnis)

savename = 'prediction_'+algo+'.xlsx'
#savename =  'prediction_'+'EnsemblesoftWithWeight'+'.xlsx'
pathsave = os.path.dirname(os.getcwd())+'/../reports/'+savename
ergebnis.to_excel(pathsave, sheet_name = 'Sheet_name_1')