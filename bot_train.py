import os
import random
import numpy as np
from collections import deque
import bot_file as bot
import skimage
import state_train as state
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Activation, Flatten
from keras.optimizers import Adam
from playng import Play
import joblib
import time




def process(input):
    # convert the input from rgb to grey
    image = skimage.color.rgb2gray(input)
    # resize image to 80x80 from 288x404
    image = skimage.transform.resize(image, (80, 80), mode='constant')
    # return image after stretching or shrinking its intensity levels
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 255))
    # scale down pixels values to (0,1)
    image = image / 255.0
    return image


def build_network():
    num_actions = 2  # number of valid actions
    print("Initializing model ....")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding='same', strides=(4, 4), input_shape=(80, 80, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (4, 4), padding='same', strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(num_actions))

    if os.path.exists("bot_model.h5"):
        print("Loading weights from bot_model.h5 .....")
        model.load_weights("bot_model.h5")
        print("Weights loaded successfully.")
    adam = Adam(lr=1e-4)
    model.compile(loss='mse', optimizer=adam)
    print("Finished building model.")

    return model


def train_network(model, mode):

    discount = 0.99  # decay rate of past observations
    observe = 3200  # timesteps to observe before training
    explore = 3000000  # frames over which to anneal epsilon
    FINAL_EPSILON = 0.0001  # final value of epsilon
    INITIAL_EPSILON = 0.1  # starting value of epsilon
    replay_memory = 50000  # number of previous transitions to remember

    if mode == 'Run':
        train = False
    elif mode == 'Train':
        train = True

    if train:
        epsilon = INITIAL_EPSILON
    else:
        epsilon = FINAL_EPSILON

    SVMclf = joblib.load("score_SVMclf.pkl")
    state_model, IMG_SIZE = state.build_network()
    sfile = open("scores_dqn.txt", "a+")
    episode = 1
    timestep = 0
    loss = 0

    # store the previous observations in replay memory
    replay = deque()

    image, score, reward, alive = Play(SVMclf, state_model, IMG_SIZE, 0)

    # preprocess the image and stack to 80x80x4 pixels
    image = process(np.array(image))
    before_image = np.stack((image, image, image, image), axis=2)
    before_image = before_image.reshape(1, before_image.shape[0], before_image.shape[1], before_image.shape[2])

    while (True):
        print('begin')
        start = time.time()
        # get an action epsilon greedy policy
        a = random.random()
        if a <= epsilon:
            action = bot.botAgent(random.randint(0,1))
        else:
            q = model.predict(before_image)
            action = bot.botAgent(np.argmax(q))
        # decay epsilon linearly
        if epsilon > FINAL_EPSILON and timestep > observe:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / explore
        # take selected action and get resultant state
        image, score, reward, alive = Play(SVMclf, state_model, IMG_SIZE, reward)

        # preprocess the image and stack to 80x80x4 pixels
        image = process(np.array(image))
        image = image.reshape(1, image.shape[0], image.shape[1], 1)  # 1x80x80x1
        after_image = np.append(image, before_image[:, :, :, :3], axis=3)

        if train:
            # add current transition to replay buffer
            replay.append((before_image, action, reward, after_image, alive))
            if len(replay) > replay_memory:
                replay.popleft()

            if timestep > observe:
                # sample a minibatch of size 32 from replay memory
                minibatch = random.sample(replay, 32)
                s, a, r, s1, alive = zip(*minibatch)
                s = np.concatenate(s)
                s1 = np.concatenate(s1)
                targets = model.predict(s)
                targets[range(32), a] = r + discount * np.max(model.predict(s1), axis=1) * alive
                loss += model.train_on_batch(s, targets)

        before_image = after_image
        timestep = timestep + 1
        end = time.time()
        print('all: ', (end - start))

        if train:
            # save the weights after every 1000 timesteps
            if timestep % 1000 == 0:
                model.save_weights("bot_model.h5", overwrite=True)
            print("TIMESTEP: " + str(timestep) + ", EPSILON: " + str(epsilon) + ", ACTION: " + str(
                action) + ", REWARD: " + str(reward) + ", Loss: " + str(loss))
            loss = 0
        elif not alive:
            print("EPISODE: " + str(episode) + ", SCORE: " + str(score))
            sfile.write(str(score) + "\n")
            episode += 1




if __name__ == '__main__':

    model = build_network()
    train_network(model, 'Train')
