#Modeling - Neural Network (Clustering)
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.cross_validation import train_test_split #split the data into train and test data
data=pd.read_csv('English11_PostProcessing.csv')
n1=len(data)
l=list(data[0:0])
l1=l.index('Q. 1 /5.00')

def model_NN(qs,qe,st):
    NN1=pd.DataFrame(data.iloc[0:n1,qs:qe])
    NN2=pd.DataFrame(data.iloc[0:n1,st])
    X_train, X_test, y_train, y_test = train_test_split(NN1, NN2, test_size = 0.25, random_state = 0)
    clf = MLPClassifier(solver='adam', alpha=1e-5,hidden_layer_sizes=(5,2), random_state=1)
    clf.fit(X_train,y_train.values.ravel()) 
    test=clf.predict(X_test)
    kfold = model_selection.KFold(n_splits=15)
    accuracy = model_selection.cross_val_score(clf,X_train,y_train.values.ravel(), cv=kfold)
    return test,accuracy.mean()*100

vocab=model_NN(l1,l1+9,l1+48)
print("Vocabulary Section: ",vocab[0])
print("accuracy Vocabulary: ",vocab[1])
grammar=model_NN(l1+9,l1+18,l1+49)
print("Grammar Section",grammar[0])
print("accuracy Vocabulary: ",grammar[1])
reading=model_NN(l1+18,l1+40,l1+50)
print("Reading Section",reading[0])
print("accuracy Reading: ",reading[1])
Computer=model_NN(l1+40,l1+45,l1+51)
print("Computer Section",Computer[0])
print("accuracy Computer: ",Computer[1])
writing=model_NN(l1+45,l1+48,l1+52)
print("Writing Section",writing[0])
print("accuracy Writing: ",writing[1])