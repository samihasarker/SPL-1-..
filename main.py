import numpy as np
import math
data=np.genfromtxt('data.csv',delimiter=',',dtype=float)
#print(data)
print(data.shape)
#print(data[0])
#print(data[0]-data[2])
'''distance = 0
for i in range(0,int(data.shape[0])-1):
    for k in range(0,int(data.shape[1])):
        distance+=pow(data[0][k]-data[i+1][k],2)
    distance=math.sqrt(distance)
    print(distance)'''