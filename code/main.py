import pandas as pd
from knn import KNN
from decisiontree import Tree
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score


file = "wine.csv"
algo = 'DT' #'DT', 'NB'

data = pd.read_csv(file)

if algo=='knn':
    KNN(data)
elif algo=='DT':
    X,Y = data.values[:,:-1],data.values[:,-1]
    if file.startswith('iris'):
        Y[Y=='setosa'] = 1
        Y[Y=='versicolor'] = 2
        Y[Y=='virginica'] = 3
    Y = Y.astype('int32')
    x1,x2,y1,y2 = train_test_split(X,Y,test_size=.2)
    
    tree  = Tree()
    tree = tree.fit(x1,y1)
    yp = np.array(tree.predict(x2))
    print(accuracy_score(y2,yp))
elif algo=='NB':
    data=read_file()
    cross_validation(data)
