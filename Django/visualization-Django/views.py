from django.shortcuts import render
from django.http import HttpResponse

def index(request):
	import seaborn as sns
	import pandas as pd
	import numpy as np
	from matplotlib import pyplot as plt
	data=pd.read_csv('English11_PostProcessing.csv')
	n1=len(data)
	l=list(data[0:0])
	l1=l.index('Surname')

	list1=list(data.iloc[:,l1+4])
	ip=input("Enter the user mailID: ")
	if(ip in list1):
		index1=list1.index(ip)
		uname=data.iloc[index1,l1+1]
		Vocabulary=data.iloc[index1,l1+58]
		Grammar=data.iloc[index1,l1+59]
		Reading=data.iloc[index1,l1+60]
		Computer=data.iloc[index1,l1+61]
		Writing=data.iloc[index1,l1+62]
	else:
		print("EmailID is not valid.")	

	vocab="<div>User Name: ",uname,"<center><div>Vocabulary Recommendation: ",Vocabulary,"<center><br><div>Grammar Recommendation: ",Grammar,"<center><br><div>Reading Recommendation: ",Reading,"<center><br><div>Computer Recommendation: ",Computer,"<center><br><div>Writing Recommendation: ",Writing

	return HttpResponse(vocab)
