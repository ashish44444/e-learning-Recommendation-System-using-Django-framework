#Find maximum of cluster
import operator
import pandas as pd
import numpy as np
data=pd.read_csv('English11_OutputToKMeans.csv')
n1=len(data)
l=list(data[0:0])
l1=l.index('Q. 1 /5.00')

def sum_Max(n,m):
	sum0=[]
	sum1=[]
	sum2=[]
	total=pd.DataFrame(data.iloc[0:n1,m])
	cluster=pd.DataFrame(data.iloc[0:n1,n])
	for i in range(n1):
		if((cluster.iloc[i].values)==0):
			sum0.append(total.iloc[i].values)
		elif((cluster.iloc[i].values)==1):
			sum1.append(total.iloc[i].values)
		else:
			sum2.append(total.iloc[i].values)   
	min_cluster={0:int(min(sum0)),1:int(min(sum1)),2:int(min(sum2))}
	min_cluster_sorted=dict(sorted(min_cluster.items(), key=operator.itemgetter(1)))
	min_cluster_sorted_list=list(min_cluster_sorted.keys())
	cluster_name={min_cluster_sorted_list[0]:"Basic",min_cluster_sorted_list[1]:"Intermediate",min_cluster_sorted_list[2]:"Advanced"}
	cluster_name_list=list(cluster_name.keys())
	cluster_name_list1=list(cluster_name.values())
	list1=[]
	for j in range(n1):
		for k in range(3):
			if((cluster.iloc[j].values)==cluster_name_list[k]):
				list1.append(cluster_name_list1[k])
				break
	return list1

data["Vocab Label"]=sum_Max(l1+43,l1+38)
data["Grammar Label"]=sum_Max(l1+44,l1+39)
data["Reading Label"]=sum_Max(l1+45,l1+40)
data["Computer Label"]=sum_Max(l1+46,l1+41)
data["Writing Label"]=sum_Max(l1+47,l1+42)

data.to_csv('English11_PostProcessing.csv',sep=',')
