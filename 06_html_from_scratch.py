import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    def fit(self, fit_data):
        self.data = fit_data
        # {||w||: [w, b]}
        opt_dict = {}
        transforms = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
        all_data = np.array([[point for point in points] for points in self.data.itervalues()])
        all_data = all_data.reshape(-1, 2)
        self.max_x_value = np.max(all_data)
        self.min_x_value = np.min(all_data)
        del all_data
        step_sizes = [self.max_x_value * .1,
                      self.max_x_value * .01,
                      self.max_x_value * .001]
        b_range_multiple, b_multiple = 5, 5
        latest_optimum = self.max_x_value * 10
        for step in step_sizes:
            w = np.array([latest_optimum, latest_optimum])
            optimized = False
            while not optimized:
                for b in np.arange(-1*self.max_x_value*b_range_multiple,
                                   self.max_x_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        for yi, points in self.data.iteritems():
                            for xi in points:
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                    break
                            if not found_option:
                                break
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]
                if w[0] < 0:
                    optimized = True
                else:
                    w = w - step
            norms = sorted([n for n in opt_dict])
            opt_choice = opt_dict[norms[0]]
            self.w, self.b = opt_choice[0], opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2

    def predict(self, features):
        x = np.array(features)
        classification = np.sign(np.dot(x, self.w) + self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=100, marker='*', color=self.colors[classification])
        return classification

    def plot_things(self):
        [[self.ax.scatter(x[0], x[1], s=50, color=self.colors[i]) for x in data_dict[i]] for i in self.data]

        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        data_range = (self.min_x_value * 0.9, self.max_x_value * 1.1)
        hyp_x_min, hyp_x_max = data_range[0], data_range[1]
        colors = {-1: 'k', 0: 'r--', 1: 'k'}
        for des_point in [-1, 0, 1]:
            y1 = hyperplane(hyp_x_min, self.w, self.b, des_point)
            y2 = hyperplane(hyp_x_max, self.w, self.b, des_point)
            self.ax.plot([hyp_x_min, hyp_x_max], [y1, y2], colors[des_point])
        plt.show()

data_dict = {-1: np.array([[1, 7],
                           [2, 8],
                           [3, 8]]),
             1: np.array([[5, 1],
                          [6, -1],
                          [7, 3]])}

svm = Support_Vector_Machine()
svm.fit(data_dict)
predict_us = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [6, -5], [5, 8]]
for p in predict_us:
    svm.predict(p)
svm.plot_things()
