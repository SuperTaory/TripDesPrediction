from tensorflow.keras.layers import Dense, Softmax, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers, losses, metrics
from sklearn.model_selection import train_test_split
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


def fetch_data(path, num=-1):
    data = []
    label = []
    ot = []
    with open(path, 'r') as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        for line in lines[:num]:
            fields = line.split(":")
            features = fields[1]
            target = int(fields[2])
            ot.append(int(fields[3]))

            feature = features.split('#')
            if len(feature) != 5:
                raise ValueError("Feature loss")
            f = [list(map(float, feature[i].split(','))) for i in range(5)]
            one_line = f[0] + f[1] + f[2] + f[3] + f[4]
            data.append(one_line)
            label.append(target)
    return np.asarray(data), np.asarray(label), ot


def main():
    # set GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    X, y, ot = fetch_data("../data/features")
    y = to_categorical(y, num_classes=166)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test, _, ot_test = train_test_split(X, y, ot, test_size=0.1)
    # print(X_train.shape, y_train.shape)

    input_1 = Input(shape=X.shape[1:])
    dense = Dense(128, activation='relu')(input_1)
    dense_1 = Dense(64, activation='relu')(dense)
    dense_2 = Dense(64, activation='relu')(dense_1)
    dense_3 = Dense(166, activation='relu')(dense_2)
    output = Softmax()(dense_3)
    model = Model(inputs=input_1, outputs=output)

    model.summary()

    my_optimizer = optimizers.Adam(learning_rate=1e-5)
    my_loss = losses.CategoricalCrossentropy()

    model.compile(optimizer=my_optimizer,
                  loss=my_loss,
                  metrics=[metrics.categorical_accuracy])

    history = model.fit(X_train, y_train,
                        batch_size=512,
                        validation_split=0.2,
                        shuffle=True,
                        epochs=100,
                        verbose=2)

    plt.title('Model loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()

    plt.title('Model accuracy')
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    pred = model.predict(x=X_test)
    pred_argmax = np.argmax(pred, axis=1)
    true_argmax = np.argmax(y_test, axis=1)
    count = sum([1 if pred_argmax[i] == true_argmax[i] else 0 for i in range(len(pred))])
    print("\nTest samples: %d, accuracy: %.2f%%" % (len(pred), count / len(pred) * 100))

    ot_acc, ot_num = [0] * 12, [0] * 12
    for i in range(len(pred)):
        ot_num[ot_test[i]] += 1
        if pred_argmax[i] == true_argmax[i]:
            ot_acc[ot_test[i]] += 1
    ot_acc = [round(ot_acc[i] / ot_num[i], 4) if ot_num[i] else 0 for i in range(len(ot_num))]
    print("ot-acc distribution")
    print(ot_acc)


if __name__ == '__main__':
    main()
