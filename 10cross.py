import pandas as pd
import math
import random
from random import shuffle
random.seed(42)

def accuracy(actual, predicted):
    acc = 0
    for i in range(0, len(actual)):
        if actual[i] == predicted[i]:
            acc += 1
    return acc / len(actual)


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def getNeighbors(trainingSet, testInstance, k):
    length = len(testInstance) - 1 
    trainingSet.sort(key=lambda x: euclideanDistance(testInstance,x, length))
    return trainingSet[0:k]

def getResponse(neighbors):
    classVotes = {}
    for x in neighbors:
        target = x[-1]
        if target in classVotes:
            classVotes[target] += 1
        else:
            classVotes[target] = 1
    max_class = max(classVotes.items(), key=lambda x: x[1])
    return max_class[0]


data = pd.read_csv("wine.csv")
N, M = data.shape
data = list(data.values)
K = 10
fold_size = N // K
print(fold_size)
start_ind = 0
avg_res = []
for i in range(K):
    shuffle(data)
    test_set = data[0:fold_size]
    train_set = data[fold_size:]

    predictions = []
    k = 10
    for x in range(len(test_set)):
        neighbors = getNeighbors(train_set, test_set[x], k)
        result = getResponse(neighbors)
        predictions.append(result)

    actual = []
    for x in range(len(test_set)):
        actual.append(test_set[x][-1])

    result = accuracy(actual, predictions)
    avg_res.append(result)
    print("Fold ", i, result)

print("Final Result:", sum(avg_res) / len(avg_res))
