#Modeling - SVM
import pandas as pd
from sklearn import svm #Support Vector Machine model
from sklearn import model_selection #k-fold validation
from sklearn.cross_validation import train_test_split #split the data into train and test data
data=pd.read_csv('English_outputpreprocessing.csv')
n1=len(data)
l=list(data[0:0])
l1=l.index('Q. 1 /5.00')

def model_svm(qs,qe,st): #Modelling & Accuracy
    svm1=pd.DataFrame(data.iloc[0:n1,qs:qe])
    svm2=pd.DataFrame(data.iloc[0:n1,st])
    X_train, X_test, y_train, y_test = train_test_split(svm1, svm2, test_size = 0.25, random_state = 0)
    clf = svm.SVC()
    clf.fit(X_train, y_train.values.ravel()) 
    test=clf.predict(X_test)
    kfold = model_selection.KFold(n_splits=15)
    accuracy = model_selection.cross_val_score(clf,X_train,y_train.values.ravel(), cv=kfold)
    return test,accuracy.mean()*100

vocab=model_svm(l1,l1+9,l1+38)
print("Vocabulary Section: ",vocab[0])
print("accuracy Vocabulary: ",vocab[1])
grammar=model_svm(l1+9,l1+18,l1+39)
print("Grammar Section",grammar[0])
print("accuracy Vocabulary: ",grammar[1])
reading=model_svm(l1+18,l1+30,l1+40)
print("Reading Section",reading[0])
print("accuracy Reading: ",reading[1])
Computer=model_svm(l1+30,l1+35,l1+41)
print("Computer Section",Computer[0])
print("accuracy Computer: ",Computer[1])
writing=model_svm(l1+35,l1+38,l1+42)
print("Writing Section",writing[0])
print("accuracy Writing: ",writing[1])
