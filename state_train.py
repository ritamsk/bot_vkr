from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
from random import shuffle
from tqdm import tqdm
import cv2
import numpy as np

TEST_FILE = "test_file.txt"
MODEL_FILE = "state_model.h5"
IMG_SIZE = [50, 50]
TRAIN_DIR = 'images'
TEST_DIR = 'images/test'

def build_network():

    global IMG_SIZE

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=[IMG_SIZE[0], IMG_SIZE[1], 1], activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.0001),
                  metrics=['accuracy'])

    if os.path.exists(''.format(MODEL_FILE)):
        model.load_weights(MODEL_FILE)
        print('weights loaded!')

    return model, IMG_SIZE

def label_img(folder):
    word_label = folder

    if word_label == 'gameplay': return [1]

    elif word_label == 'rip': return [0]


def create_train_data():
    global IMG_SIZE
    training_data = []
    for folder in tqdm(os.listdir(TRAIN_DIR)):
        if folder!='test':
            path = os.path.join(TRAIN_DIR, folder)
            for img in tqdm(os.listdir(path)):
                label = label_img(folder)
                path1 = os.path.join(path,img)
                img = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
                training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)

    return training_data


def create_test_data():
    global IMG_SIZE
    testing_data = []
    for folder in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, folder)
        for img in tqdm(os.listdir(path)):

            path1 = os.path.join(path, img)
            img_num = label_img(folder)
            img = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE[0], IMG_SIZE[1]))
            testing_data.append([np.array(img), img_num])

    shuffle(testing_data)

    return testing_data

def model_train():
    model, IMG_SIZE = build_network()
    # Training
    train_data = create_train_data()
    train = train_data[:100]
    test = train_data[:30]

    X = np.array([i[0] for i in train]).reshape(100, IMG_SIZE[0], IMG_SIZE[1], 1)
    Y = np.array([i[1] for i in train]).reshape(100, 1)

    test_x = np.array([i[0] for i in test]).reshape(30, IMG_SIZE[0], IMG_SIZE[1], 1)
    test_y = np.array([i[1] for i in test]).reshape(30, 1)


    model.fit(X, Y, epochs=7, validation_data=(test_x, test_y), steps_per_epoch=5, validation_steps=5)
    model.save_weights(MODEL_FILE)

    # Testing
    test_data = create_test_data()
    for num, data in enumerate(test_data[:12]):
        img_num = data[1]
        img_data = data[0]
        print(img_num)
        data = img_data.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
        model_out = model.predict([data])[0]

        print(model_out)


#model_train()