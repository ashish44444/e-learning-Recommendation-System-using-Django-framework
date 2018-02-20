import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
df=pd.read_csv('English11_PostProcessing.csv')
n1=len(df)
l=list(df[0:0])
l1=l.index('Q. 1 /5.00')

ax=df.iloc[0:n1,l1+38:l1+43]
sns.boxplot(data=ax)
plt.show()
