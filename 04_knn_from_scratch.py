import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings

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
    print(count_votes)
    vote_result = count_votes[0][0]
    return vote_result

result = k_nearest_neighbors(dataset, new_point, k=5)
print(result)

[[plt.scatter(*point, s=100, color=cls) for point in points] for cls, points in dataset.iteritems()]
plt.scatter(*new_point, s=100, c=result)
plt.show()
