#Overall visualization
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
data=pd.read_csv('English11_PostProcessing.csv')
n1=len(data)
l=list(data[0:0])
l1=l.index('Q. 1 /5.00')


def count(n,n1):
    count_Basic=0
    count_Intermediate=0
    count_Advanced=0
    for i in range(n1):
        if(data.iloc[i,n]=='Basic'):
            count_Basic=count_Basic+1
        elif(data.iloc[i,n]=='Intermediate'):
            count_Intermediate=count_Intermediate+1
        else:
            count_Advanced=count_Advanced+1
    return count_Basic,count_Intermediate,count_Advanced

vocab=count(l1+48,n1)
grammar=count(l1+49,n1)
reading=count(l1+50,n1)
comp=count(l1+51,n1)
writing=count(l1+52,n1)

n_groups = 5
index = np.arange(n_groups)
bar_width = 0.2
opacity = 0.8

basic=[vocab[0],grammar[0],reading[0],comp[0],writing[0]]
intermediate=[vocab[1],grammar[1],reading[1],comp[1],writing[1]]
advanced=[vocab[2],grammar[2],reading[2],comp[2],writing[2]]

rects1 = plt.bar(index,basic, bar_width,alpha=opacity,color='b',label='Basic')
rects2 = plt.bar(index+bar_width,intermediate, bar_width,alpha=opacity,color='g',label='Intermediate')
rects3 = plt.bar(index+bar_width+bar_width,advanced, bar_width,alpha=opacity,color='r',label='Advanced')

plt.xlabel('Sections')
plt.ylabel('No. of Students')
plt.title('Overall Student Label')
plt.xticks(index + bar_width, ('Vocab', 'Grammar', 'Reading','Computer','Writing'))
plt.legend()
 
plt.tight_layout()
plt.show()
