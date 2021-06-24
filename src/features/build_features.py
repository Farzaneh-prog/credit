import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Preprocessor:
    def __init__(self, data):
        # hier hab ich daten bekommen
        
        data = data.sample(frac=1)
        #Shuffle data

        if (data.isna().sum().sum() != 0) or (data.isnull().sum().sum() != 0) or (data.isnull().values.any() == True):
            self.Daten_Check = 'Fehler! Die Daten sollen bearbeitet werden.'
        else :
            self.Daten_Check = 'Die Daten sind in Ordnung.'
         # hier werden die Vollständigkeit der Daten gecheckt

        print(self.Daten_Check)  
        
        #data_conv = pd.get_dummies(data, drop_first=True, dtype=float)
        # hier werden die Binäre ja,nein zum 0 und 1 convertiert

        self.X = data.drop(['kredit'], axis = 1)
        self.y = data['kredit']
        
        #Statistics
        #Get the absolute number of how many instances in our data belong to class Positive
        self.count_positive = len(data.loc[data['kredit']== 1.0])
        
        #Get the absolute number of how many instances in our data belong to class one
        self.count_negative = len(data.loc[data['kredit']== 0.0])


  
    def get_data(self, alpha):
        #alpha = 0.2
        # hier werden daten gesplittet
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size=alpha, random_state=14)
        # ab hier hab ich jetzt den scaler fertig
        return self.X_train, self.X_test, self.y_train, self.y_test, self.X, self.y,
        
    def statistica(self):
        
        #Get the relative number of how many instances in our data belong to class zero
        percentage_positive = round(self.count_positive/(self.count_negative + self.count_positive),4)*100

        #Get the relative number of how many instances in our data belong to class one
        percentage_negative = round(self.count_negative/(self.count_positive + self.count_negative),4)*100

        return self.count_positive, self.count_negative, percentage_positive, percentage_negative

