import numpy as np
import random
import time
import matplotlib.pyplot as plt
from NNFull import NeuralNetwork as NN

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.xlim(0, 1)
plt.ylim(0, 1)

targetPoint, = ax.plot(0, 0, 'x')
positionPoint, = ax.plot(0, 0, 'o')

layer_sizes = [6, 24, 12, 1]
layer_functions = ['relu', 'relu', 'linear']
qlearn = NN(layer_sizes, layer_functions)


def random_position():
    return np.random.uniform(size=2)


def distance(a, b):
    total = np.power(a-b, 2)
    total = np.sum(total)
    return np.sqrt(total)

learning_rate = 0.1
gamma = 0.99
exp_decay = 0.001
min_exp = 0.01
speed = 0.05


episode = 0
while True:
    print(f'Starting episode {episode}')
    target = random_position()
    targetPoint.set_data(target)
    fig.canvas.draw()
    fig.canvas.flush_events()

    position = random_position()

    exp_rate = max(min_exp, np.exp(-exp_decay * episode))


    iteration = 0
    while iteration < 100 and distance(target, position) > 0.1:
        iteration += 1
        time.sleep(0.1)
        if np.random.uniform() < exp_rate:
            # Explore
            action = np.random.randint(0, 4)
        else:
            # Exploit
            difference = (target - position)
            max_q = -np.inf
            action = 0
            for pos_action in range(4):
                action_input = np.zeros(4)
                action_input[pos_action] = 1
                inputs = np.concatenate([difference, action_input])
                q_value = qlearn.feed_forward(inputs)[0]
                if q_value > max_q:
                    q = max_q
                    action = pos_action
        # Now we have the action we'll use
        old_position = position.copy()
        match action:
            case 0:
                position[1] += speed
            case 1:
                position[0] += speed
            case 2:
                position[0] -= speed
            case 3:
                position[1] -= speed

        new_dist = distance(position, target)
        reward = (2 - new_dist)

        difference = (target - position)
        next_max_q = -np.inf
        for pos_action in range(4):
            action_input = np.zeros(4)
            action_input[pos_action] = 1
            inputs = np.concatenate([difference, action_input])
            q_value = qlearn.feed_forward(inputs)[0]
            next_max_q = max(next_max_q, q_value)


        difference = (target - old_position)
        action_input = np.zeros(4)
        action_input[action] = 1
        inputs = np.concatenate([difference, action_input])
        old_q_value = qlearn.feed_forward(inputs)

        target = (1 - learning_rate) * old_q_value + learning_rate * (reward + gamma * next_max_q)
        qlearn.train(inputs, target)

        positionPoint.set_data(position)
        fig.canvas.draw()
        fig.canvas.flush_events()

    episode += 1
