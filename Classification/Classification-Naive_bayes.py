#Modeling - Naive Bayes
import pandas as pd
from sklearn.naive_bayes import GaussianNB  #Naive Bayes model
from sklearn import model_selection #k-fold validation
from sklearn.cross_validation import train_test_split #split the data into train and test data
data=pd.read_csv('English_outputpreprocessing.csv')
n1=len(data)
l=list(data[0:0])
l1=l.index('Q. 1 /5.00')

def model_NB(qs,qe,st): #Modelling & Accuracy
    NB1=pd.DataFrame(data.iloc[0:n1,qs:qe]) #particular section data
    NB2=pd.DataFrame(data.iloc[0:n1,st]) #particular section label
    X_train, X_test, y_train, y_test = train_test_split(NB1, NB2, test_size = 0.25, random_state = 0) #Training & Testing data
    clf = GaussianNB() #Gaussian Naive Bayes model
    clf.fit(X_train,y_train.values.ravel())  #fit our data into model
    test=clf.predict(X_test) #predict the test data
    kfold = model_selection.KFold(n_splits=15) #k-fold cross validation
    accuracy = model_selection.cross_val_score(clf,X_train,y_train.values.ravel(), cv=kfold) # find accuracy
    return test,accuracy.mean()*100

vocab=model_NB(l1,l1+9,l1+38)
print("Vocabulary Section: ",vocab[0])
print("accuracy Vocabulary: ",vocab[1])
grammar=model_NB(l1+9,l1+18,l1+39)
print("Grammar Section",grammar[0])
print("accuracy Vocabulary: ",grammar[1])
reading=model_NB(l1+18,l1+30,l1+40)
print("Reading Section",reading[0])
print("accuracy Reading: ",reading[1])
Computer=model_NB(l1+30,l1+35,l1+41)
print("Computer Section",Computer[0])
print("accuracy Computer: ",Computer[1])
writing=model_NB(l1+35,l1+38,l1+42)
print("Writing Section",writing[0])
print("accuracy Writing: ",writing[1])
