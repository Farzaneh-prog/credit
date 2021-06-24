import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import sys, os
sys.path.append("..")

def pl_ROC(algo, roc_auc, fpr, tpr):

    #change the working directory
    path_start = os.getcwd()
    pathr=os.path.dirname(os.getcwd())+'/reports/figures'
    os.chdir(pathr)
    
    #define figure size 
    plt.figure(figsize=(12,12))

    #add title
    plt.title('{} ROC Curve'.format(algo))

    # plot and add labels to plot
    plt.plot(fpr, tpr, 'b', label = 'Kredit data: {} AUC =  {}'.format(algo,(round(roc_auc,4))))
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
        
    pltname = "ROC_Curve_"+algo+".png"
    plt.savefig(pltname)
   
    #plt.show()

    #change to the start working directory
    os.chdir(path_start)
    return

def Diabetes_tree(best_model, X):
    plt.figure(figsize=(30,14))
    tree.plot_tree(best_model, filled=True, fontsize=12)
    
    #change the working directory
    path_start = os.getcwd()
    pathr=os.path.dirname(os.getcwd())+'/reports/figures'
    os.chdir(pathr)
    export_graphviz(best_model,out_file=("Kredit_tree.dot"), feature_names=X.columns[:],class_names=(['bad Credit','good Credit']), rounded=True, filled=True)
    os.system("dot -Tpng Kredit_tree.dot -o Kredit_tree.png") 
    os.system("dot -Tps Kredit_tree.dot -o Kredit_tree.ps")

    #change to the start working directory
    os.chdir(path_start)
    return

def Akkuranz_pl(Akkuranz, algo):

    #change the working directory
    path_start = os.getcwd()
    pathr=os.path.dirname(os.getcwd())+'/reports/figures'
    os.chdir(pathr)

    #define figure size 
    plt.figure(figsize=(8,8))

    #add title
    plt.title('{} accuracy versus train/test data size ratio'.format(algo))

    # plot and add labels to plot
    plt.plot(np.linspace(0.2,0.8,7), Akkuranz, 'b--*', label = 'K_fold Accuracy with best parameters')
    plt.legend(loc = 'lower right')
    plt.xlabel('test/train ratio')
    plt.ylabel('K_fold Accuracy')
        
    pltname = "AccuracySize_"+algo+".png"
    plt.savefig(pltname)
    #savefig options: dpi = 96pxl, transparent= True, bbox_inches='tight', pad_inches

    #plt.show()

    #change to the start working directory
    os.chdir(path_start)

def ConfmatPl(cm, alg, vot):
    #Nam = []
    #Nam = alg.append(vot)
    
    #Nam = ["DT","rf","knn","svm",'hard','soft','softWithWeight','softCutoff','softWWCutoff']
    Nam = ["DT","rf","knn",'hard','soft','softWithWeight','softCutoff','softWWCutoff']
    
    #change the working directory
    path_start = os.getcwd()
    pathr=os.path.dirname(os.getcwd())+'/reports/figures'
    os.chdir(pathr)

    #define figure size 
    #axarr = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12, 8))
    
    #add title
    #axarr.set_title(Nam)
    for i in range(0,len(cm)):
    #define figure size 
        plt.figure(figsize=(4,4))

        cm_display = ConfusionMatrixDisplay(cm[i]).plot()

        pltname = "ConfMat"+Nam[i]+".png"
        plt.savefig(pltname)

    #change to the start working directory
    os.chdir(path_start)
