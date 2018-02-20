import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df=pd.read_csv('English11_PostProcessing.csv')

ax=df.iloc[0]
vocab=(ax[50]/115)*100
grammar=(ax[51]/115)*100
reading=(ax[52]/60)*100
comp=(ax[53]/10)*100
writing=(ax[54]/50)*100

bar_width = 0.2
n_groups = 5
opacity = 0.8
index = np.arange(n_groups)

basic=[vocab,grammar,reading,comp,writing]
rects1 = plt.bar(index,basic, bar_width,alpha=opacity,color='b',label='Section Percentage')
plt.xlabel('Sections')
plt.ylabel('Student Percentage')
plt.title('Overall Student Label')
plt.xticks(index, ('Vocab', 'Grammar', 'Reading','Computer','Writing'))
plt.legend()

plt.tight_layout()
plt.show()