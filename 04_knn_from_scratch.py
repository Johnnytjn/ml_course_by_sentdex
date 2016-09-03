from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
from random import shuffle

style.use('fivethirtyeight')

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_point = [4, 4]


def euclidian_distance(features, predict, use_np=True):
    if use_np:
        return np.linalg.norm(np.array(features) - np.array(predict))
    else:
        return np.sqrt(np.sum((np.array(features) - np.array(predict)) ** 2))


def k_nearest_neighbors(data, predict, k=3):
    if len(data) > k:
        warnings.warn("K is less then total amount of classes")
    distances = []
    for group in data:
        for features in data[group]:
            eucl_dist = euclidian_distance(features, predict)
            distances.append([eucl_dist, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    count_votes = Counter(votes).most_common(1)
    confidance = count_votes[0][1] / k
    vote_result = count_votes[0][0]
    return vote_result, confidance

# result = k_nearest_neighbors(dataset, new_point, k=5)
# print(result)
#
# [[plt.scatter(*point, s=100, color=cls) for point in points] for cls, points in dataset.iteritems()]
# plt.scatter(*new_point, s=100, c=result)
# plt.show()

df = pd.read_csv('breast-cancer-wisconsin.data.txt', index_col=0)
df.replace('?', -99999, inplace=True)

full_data = df.astype(float).values.tolist()
shuffle(full_data)
test_size = int(0.3*len(full_data))
train_set = {2: [], 4: []}
test_set = {2: [], 4: []}
train_data = full_data[:-test_size]
test_data = full_data[-test_size:]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct, total = 0, 0
for group in test_set:
    for data in test_set[group]:
        vote, conf = k_nearest_neighbors(train_set, data, k=5)
        total += 1
        correct += 1 if group == vote else 0
print("Accuracy: ", correct/total)
