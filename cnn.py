import os

import cv2 as cv
import keras
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

nomes_classes = ["flooded", "no_flooded"]


def load_data_path(path, x, y):
    for img_path in os.listdir(path):
        if img_path.find("_1") < 0:
            y.append(0)
        else:
            y.append(1)
        img = cv.imread(f'{path}/{img_path}', cv.IMREAD_GRAYSCALE)
        x.append(cv.resize(img, (512, 360)) / 255.0)


def load_dataset():
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    load_data_path("data/train", x_train, y_train)
    load_data_path("data/test", x_test, y_test)

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load_dataset()

    nn = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(512, 360, 1)),
        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(128, 5, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(4),
        keras.layers.Conv2D(128, 5, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(4),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation="sigmoid")
    ])

    print(nn.summary())

    x_train_new = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_test_new = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    nn.compile(optimizer='adam',
               loss=keras.losses.binary_crossentropy,
               metrics=['accuracy'])

    history_nn = nn.fit(x_train_new, y_train, epochs=25, validation_data=(x_test_new, y_test))

    pd.DataFrame(history_nn.history).plot(figsize=(12, 8))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()

    result = nn.predict(x_test[:60])

    for i in range(len(result)):
        print(f'Previsao: {np.argmax(result[i], axis=-1)} | Verdadeiro: {y_test[i]}')
