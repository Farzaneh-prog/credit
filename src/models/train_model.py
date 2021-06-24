import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score, GridSearchCV,  KFold 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.svm import SVC

def train_model(X_train, X_test, y_train, y_test, algo):
    # all available models in dictionary
    models = {"DT":DecisionTreeClassifier(criterion='entropy',max_depth=5, random_state=14, class_weight='balanced'),"rf":RandomForestClassifier(n_estimators=10, max_depth=3, random_state=0, class_weight='balanced'),"knn": KNeighborsClassifier(n_neighbors = 4, weights='distance', algorithm='auto',leaf_size=2, p=2, metric='minkowski'), "svm":SVC(kernel='poly', probability = True, C = 0.1, gamma='auto', class_weight='balanced')}

    # select model
    model = models[algo]
    
    # train model
    model.fit(X_train,y_train)

    # score model
    accuracy = model.score(X_test, y_test)
    #print('Model {} successfully trained with an accuracy of {}% '.format(algo,round(accuracy,4)*100))  
    model_pred = model.predict(X_test)
    conf_mat = confusion_matrix(y_test, model_pred, normalize='all')

    # give back
    return model, conf_mat

def optimize_model(model, X_train,y_train, algo):

    para={"DT":{'criterion':['gini','entropy'],'max_depth':[i for i in range(1,6)],'min_samples_split':[i for i in range(2,20)]},"rf":{'criterion':['gini','entropy'],'n_estimators':[3, 10, 30],'max_depth':[i for i in range(1,6)],'min_samples_split':[i for i in range(2,10)]},"knn":{'n_neighbors':[i for i in range(2,10)],'weights':['uniform','distance'],'leaf_size':[i for i in range(2,30)]},"svm":{'kernel':('linear','poly','sigmoid','rbf'),'C':[0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0],'gamma':(1,2,3,'auto')}}

    #parameterraum which shall be tested to optimize hyperparameters
    model_para = para[algo]

    #GridSearchCV object
    model_grd = GridSearchCV(model, model_para, cv=5) 

    #creates differnt classifiers with all the differnet parameters out of our data
    model_grd.fit(X_train,y_train)
    #man könnte hier für Knn Methode auch auf (X,y) trainieren um dem Optimale Hyperparameter zu finden und den gerechnete Accuranz hier ist das richtige Accuranz
 
    #best paramters that were found
    model_best_parameters = model_grd.best_params_  
    print('Model {} successfully optimized with the best parameters of {}.'.format(algo,model_best_parameters))  

    #new model object with best parameters
    model_with_best_parameters = model_grd.best_estimator_

    return model_with_best_parameters

def kfoldevaluate_optimzed_model(algo, model_with_best_parameters, X, y, X_train, X_test, y_train, y_test):

    # train model
    model_with_best_parameters.fit(X_train,y_train)

    # score model
    accuracy = model_with_best_parameters.score(X_test, y_test)
    #print('Model {} successfully trained with an accuracy of {}% '.format(model_with_best_parameters,round(accuracy,4)*100))  
    model_pred = model_with_best_parameters.predict(X_test)
    conf_mat = confusion_matrix(y_test, model_pred, normalize='all')

    #k_fold object to optimize the accuracy measurement
    accuracy_k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

    #scores reached with different splits of training/test data 
    accuracy_k_fold_scores = cross_val_score(model_with_best_parameters, X, y, cv=accuracy_k_fold, n_jobs=-1)

    #arithmetic mean of accuracy scores 
    mean_accuracy_best_parameters = np.mean(accuracy_k_fold_scores)

    print('K_fold Accuracy of {} with best parameters is {}% '.format(algo, round(mean_accuracy_best_parameters, 4)*100))

    return model_with_best_parameters, conf_mat, mean_accuracy_best_parameters

def roc_vorbereitung(model_with_best_parameters, X_test,y_test):
    
    # get probabilities of class membership of test instances
    model_probs = model_with_best_parameters.predict_proba(X_test)

    #get col with positive probabilities
    y_pred_proba = model_probs[:,1]

    # get false positive rate, true positive rate and threshold values
    fpr, tpr, threshold = roc_curve(y_test, y_pred_proba, pos_label=1)
    
    # Compute Area Under the Curve (AUC) using the trapezoidal rule
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, threshold, roc_auc

def Ensembel(best_model_list, X,y, X_train, X_test, y_train, y_test, vote, method_Weight):

    #voteList = {'hard':VotingClassifier(estimators=[('DT', best_model_list[0]), ('rf', best_model_list[1]), ('knn', best_model_list[2]), ('svm', best_model_list[3])], voting='hard')
     #           ,'soft': VotingClassifier(estimators=[('DT', best_model_list[0]), ('rf', best_model_list[1]), ('knn', best_model_list[2]), ('svm', best_model_list[3])], voting='soft')
      #          ,'softWithWeight':VotingClassifier(estimators=[('DT', best_model_list[0]), ('rf', best_model_list[1]), ('knn', best_model_list[2]), ('svm', best_model_list[3])], voting='soft', weights=method_Weight, flatten_transform=True)}
    
    voteList = {'hard':VotingClassifier(estimators=[('DT', best_model_list[0]), ('rf', best_model_list[1]), ('knn', best_model_list[2])], voting='hard')
                ,'soft': VotingClassifier(estimators=[('DT', best_model_list[0]), ('rf', best_model_list[1]), ('knn', best_model_list[2])], voting='soft')
                ,'softWithWeight':VotingClassifier(estimators=[('DT', best_model_list[0]), ('rf', best_model_list[1]), ('knn', best_model_list[2])], voting='soft', weights=method_Weight, flatten_transform=True)}
    
    Ensemblemodel = voteList[vote]
    Ensemblemodel = Ensemblemodel.fit(X_train,y_train)

    #k_fold object to optimize the accuracy measurement
    accuracy_k_fold = KFold(n_splits=5, shuffle=True, random_state=0)

    #scores reached with different splits of training/test data 
    accuracy_k_fold_scores = cross_val_score(Ensemblemodel, X, y, cv=accuracy_k_fold, scoring='accuracy', n_jobs=-1)

    #arithmetic mean of accuracy scores 
    mean_accuracy_Ensembelhard = np.mean(accuracy_k_fold_scores)

    print('K_fold Accuracy of {} Voting Classifier with best parameters is {}% '.format(vote, round(mean_accuracy_Ensembelhard, 4)*100))

    model_pred = Ensemblemodel.predict(X_test)

    conf_mat = confusion_matrix(y_test, model_pred, normalize='all')

    return Ensemblemodel, conf_mat

def predict_with_cutoff(best_model_list, threshold, X_test, y_test, n):

    print("Cutoff/threshold at: " + str(threshold))
    Ensemblemodel = best_model_list[n]
    model_pred = (Ensemblemodel.predict_proba(X_test)[:,1] >= threshold).astype(bool)

    #model_pred = [1 if x >= threshold else 0 for x in Ensemblemodel.predict_proba(X_test)[:, 1]] # 1 is for natural birth, we want to maximize the true positive and minimize the falsh positive. If it is more than threshold we set it to zero
    conf_mat = confusion_matrix(y_test, model_pred, normalize='all')

    return model_pred, conf_mat
