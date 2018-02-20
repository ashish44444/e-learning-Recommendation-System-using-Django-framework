#Clustering
import pandas as pd
from sklearn.cluster import KMeans
data=pd.read_csv('English11_InputToKMeans.csv')
#data5=data5.replace('-','0',regex=True)
n1=len(data)
l=list(data[0:0])
l1=l.index('Q. 1 /5.00')

def kmeans_model(n):
    vocab1=pd.DataFrame(data.iloc[0:n1,n])
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(vocab1)
    labels = kmeans.predict(vocab1)
    centroids = kmeans.cluster_centers_
    return labels


data["Vocab Section"]=kmeans_model(l1+38)
data["Grammar Section"]=kmeans_model(l1+39)
data["Reading Section"]=kmeans_model(l1+40)
data["Computer Section"]=kmeans_model(l1+41)
data["Writing Section"]=kmeans_model(l1+42)

data.to_csv('English11_OutputToKMeans.csv',sep=',')