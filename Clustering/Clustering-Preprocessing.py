#Preprocessing - Clustering
import pandas as pd
import numpy as np
data=pd.read_csv('English11_test.csv')
data=data.replace('-','0',regex=True)
n1=len(data)

def sec_total(c1,c2):
    total_list=[]
    index=pd.DataFrame(data.iloc[0:n1,c1:c2])
    for i in range(n1):
        total_list.append(sum(index.iloc[i]))
    return total_list

data["Vocab Total"]=sec_total(11,20)
data["Grammar Total"]=sec_total(20,29)
data["Reading Total"]=sec_total(29,41)
data["Computer Total"]=sec_total(41,46)
data["Writing Total"]=sec_total(46,49)

data.to_csv('English11_InputToKMeans.csv',sep=',')        