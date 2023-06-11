import numpy as np
import matplotlib.pyplot as plt
from NNFull import NeuralNetwork as N6

import time
import random

layer_sizes = [2, 300, 30, 1]
functions = ['relu', 'relu', 'sigmoid']
nn = N6(layer_sizes)
# nn = N6(layer_sizes, functions)

training_data = [[[0, 1], [1]],
                 [[1, 0], [1]],
                 [[0, 0], [0]],
                 [[1, 1], [0]]]

iterations = 5000

c = time.time()
for i in range(iterations):
    data = random.choice(training_data)
    nn.train(data[0], data[1])
print('Time', time.time() - c)

for data in training_data:
    res = nn.feed_forward(data[0])
    print(data, res)



# def get_points(xsamples, ysamples):
#     output = np.empty_like(xsamples)
#     for i in range(len(xsamples)):
#         output[i] = nn.feed_forward([xsamples[i], ysamples[i]])[0]
#     return output
#
#
# fig = plt. figure()
# ax = fig.add_subplot(projection='3d')
# xs = np.linspace(-1, 2)
# ys = np.linspace(-1, 2)
# zs = get_points(xs, ys)
#
# xNew = np.empty(0)
# yNew = np.empty(0)
# zNew = np.empty(0)
# for x in xs:
#     for y in ys:
#         xNew = np.append(xNew, x)
#         yNew = np.append(yNew, y)
#         zNew = np.append(zNew, nn.feed_forward([x, y])[0])
# # ax.scatter(xs, ys, zs)
# ax.scatter(xNew, yNew, zNew, c=zNew, s=7)
# for point in training_data:
#     coords, label = point
#     marker = '1'
#     colour = 'orange'
#     if label[0] == 0:
#         marker = '.'
#         colour = 'green'
#     ax.scatter(coords[0], coords[1], 0.5, s=500, c=colour, marker=marker)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('NN Output')
# plt.show()

