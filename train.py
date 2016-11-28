import sys
import random
from collections import deque

import cv2
import numpy as np
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Flatten, Lambda
from keras.models import Model
from keras.optimizers import Adam
import keras.objectives
import keras.backend as K

import game.game as game

OBSERVE = 10000
EXPLORE = 3000000
GAMMA = 0.99
INITIAL_EPSILON = 0.1
FINAL_EPSILON = 0.0001
REPLAY_MEMORY = 50000
BATCH_SIZE = 32

def getModel():
    input_img = Input(shape=(4, 80, 80))
    action_mask = Input(shape=(2,))
    x = Convolution2D(32, 8, 8, border_mode='same', subsample=(4, 4),
            activation='relu', dim_ordering='th')(input_img)
    x = MaxPooling2D((2, 2), dim_ordering='th')(x)
    x = Convolution2D(64, 4, 4, border_mode='same', subsample=(2, 2), 
            activation='relu', dim_ordering='th')(x)
    x = Convolution2D(64, 3, 3, border_mode='same', 
            activation='relu', dim_ordering='th')(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(2)(x)
    Q = Lambda(lambda x: x.sum())(x)
    model = Model(input=[input_img, action_mask], output=[x, Q])
    return model

def train():
    game_state = game.GameState()

    ds = deque()

    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    img, reward, terminal = game_state.frame_step(do_nothing)
    img = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    img = np.concatenate([img[np.newaxis, :, :]] * 4, axis=0)

    num_minibatch = 0
    epsilon = INITIAL_EPSILON
    model = getModel()

    def loss_function(y_true, y_pred):
        from IPython import embed;embed()
        return keras.objectives.mean_squared_error(y_true, y_pred[1])

    adam_optimizer = Adam(lr=1e-6)
    model.compile(optimizer=adam_optimizer, loss=loss_function)

    while True:
        action_infered = model(img, np.array([0, 0]))[0]
        action = np.zeros(2)
        if random.random() < epsilon:
            action[random.randint(0, 1)] = 1
        else:
            index = np.argmax(action)
            action[index] = 1

        img_, reward_, terminal_ = game_state.frame_step(action)
        img_ = cv2.cvtColor(cv2.resize(img_, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, img_ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
        img_ = np.concatenate([img_[np.newaxis, :, :], img[:3, :, :]], axis=0)

        ds.append((img, action, reward_, img_, terminal_))
        if len(ds) > REPLAY_MEMORY:
            ds.popleft()

        if num_minibatch > OBSERVE:
            minibatch = random.sample(ds, BATCH_SIZE)

            batch_img = [d[0] for d in minibatch]
            batch_action = [d[1] for d in minibatch]
            batch_reward_ = [d[2] for d in minibatch]
            batch_img_ = [d[3] for d in minibatch]
            batch_terminal = [d[4] for d in minibatch]

            batch_Q = []
            batch_next_Q = model(np.stack(batch_img), np.zeros((BATCH_SIZE, 2)))[0]
            for i in range(len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    batch_Q.append(batch_next_Q[i])
                else:
                    batch_Q.append(batch_next_Q[i] + GAMMA * np.max(batch_next_Q[i]))

            model.train_on_batch([batch_img, batch_action], batch_Q)
            
        if num_minibatch > OBSERVE and num_minibatch < OBSERVE + EXPLORE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

        img = img_

        num_minibatch += 1

def main():
    train()
    
if __name__ == '__main__':
    main()
