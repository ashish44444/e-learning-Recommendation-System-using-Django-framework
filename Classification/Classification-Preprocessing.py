#Preprocessing
import pandas as pd #to import the dataset
data=pd.read_csv('English-inputpreprocessing.csv') # read csv data
data=data.replace('-','0',regex=True) # convert missing value to zero
n1=len(data) #length of the dataset

def process(qs,qe,max_marks): # for labeling the dataset 
    vocab =data.iloc[0:n1,qs:qe] #qs:particular section starting & qe:particular section ending 
    vocab_section=[]
    for i in range(n1):  
        count=0
        for j in range(qe-qs):
            c1=vocab.iloc[i,j]
            count=count + float(c1)        
        c=(count/max_marks)*100
        if(c>=70):
            ct='Advanced'
        elif(c>50):
            ct='Intermediate'
        else:
            ct='Basic'
        vocab_section.append(ct) 
    return vocab_section
        
data["Vocab Section"]=process(10,19,115) 
data["Grammar Section"]=process(19,28,115)
data["Reading Section"]=process(28,40,60)
data["Computer Section"]=process(40,45,10)
data["Writing Section"]=process(45,48,50)

data.to_csv('English_outputpreprocessing.csv',sep=',') #to export in csv
