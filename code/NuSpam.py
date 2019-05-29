import numpy as np
import pandas as pd
import math

def  read_file():
	data = pd.read_csv("spambase.csv")
	return data

def find_prob(train,test):
	spamdb=train[train.iloc[:,57]==1]
	hamdb=train[train.iloc[:,57]==0]
	

	total=np.zeros((54,2),dtype=int)
	hamProb=np.zeros((54))
	spamProb=np.zeros((54))
	for i in range(54):
		total[i][0]=hamdb.iloc[:,i].sum()
		total[i][1]=spamdb.iloc[:,i].sum()
	
	for i in range(54):
		hamProb[i]=total[i][0]/(total[i][0]+total[i][1])
		spamProb[i]=(1-hamProb[i])

	ham=0
	spam=0


	for i in range(0,54):

		if test[i] !=0 :
			if hamProb[i]>0:
				ham=ham+math.log(hamProb[i])
			if spamProb[i]>0:
				spam=spam+math.log(spamProb[i])

	cls=5
	if(ham>spam):
		cls=0
	else:
		cls=1
	return cls


	

def cross_validation(data):
	totalLength=len(data)
	sFactor=.9
	par=int(.1*totalLength)-1
	totalData=data.copy()
	parcent=0


	for x in range(10):
		testSegment = data.iloc[x*par:(x*par+(par-1))] 
		trainSegement = data.drop(data.index[x*par:(x*par+(par-1))])


		match=0
		for y in range(len(testSegment)):
			cls=find_prob(trainSegement, testSegment.iloc[y])
			if cls==testSegment.iloc[y][57]:
                
				match+=1


		print(match,len(testSegment))
		parcent+=match/(par*10)


	print("Result is: ",parcent*100,"%")


data=read_file()
cross_validation(data)
