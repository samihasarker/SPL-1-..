import pandas as pd
import math


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
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=lambda x: x[1])
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=lambda x: x[1], reverse=True)
    return sortedVotes[0][0]


data = pd.read_csv("iris.csv")
N, M = data.shape
data = list(data.values)
K = 10
fold_size = N // K
print(fold_size)
start_ind = 0
avg_res = []
for i in range(K):
    train_set = []
    test_set = []
    for j in range(N):
        if start_ind <= j < start_ind + fold_size:
            test_set.append(data[j])
        else:
            train_set.append(data[j])
    start_ind += fold_size

    predictions = []
    k = 1
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
