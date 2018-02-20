import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
data=pd.read_csv('English11_PostProcessing.csv')
n1=len(data)
l=list(data[0:0])
l1=l.index('Surname')

def individual_marks(i,n,max_marks):  
    l1=[]
    for i in range(n1):
        l1.append((data.iloc[:,n]/max_marks)*100)
    totMarks=(data.iloc[i,n]/max_marks)*100
    vocab=np.mean(l1)
    maxVocab=max(data.iloc[:,n]/max_marks)*100
    return totMarks,vocab,maxVocab

list1=list(data.iloc[:,l1+4])
ip=input("Enter the user mailID: ")
if(ip in list1):
    index1=list1.index(ip)
    Vocabulary=individual_marks(index1,l1+48,115)
    Grammar=individual_marks(index1,l1+49,115)
    Reading=individual_marks(index1,l1+50,60)
    Computer=individual_marks(index1,l1+51,10)
    Writing=individual_marks(index1,l1+52,50)
else:
    print("EmailID is not valid.")

bar_width = 0.2
n_groups = 5
opacity = 0.8
index = np.arange(n_groups)

user_marks=[Vocabulary[0],Grammar[0],Reading[0],Computer[0],Writing[0]]
average_marks=[Vocabulary[1],Grammar[1],Reading[1],Computer[1],Writing[1]]
highest_marks=[Vocabulary[2],Grammar[2],Reading[2],Computer[2],Writing[2]]

rects1 = plt.bar(index,user_marks, bar_width,alpha=opacity,color='b',label='User Marks')
rects2 = plt.bar(index+bar_width,average_marks, bar_width,alpha=opacity,color='g',label='Average Marks')
rects3 = plt.bar(index+bar_width+bar_width,highest_marks, bar_width,alpha=opacity,color='r',label='Highest Marks')

plt.xlabel('Sections')
plt.ylabel('Performance of Students')
plt.title('Overall Marks Details')
plt.xticks(index + bar_width, ('Vocab', 'Grammar', 'Reading','Computer','Writing'))
plt.legend()
 
plt.tight_layout()
plt.show()

