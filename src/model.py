import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers, optimizers, losses, metrics
from tensorflow.keras.layers import Layer, Input, Conv1D, Softmax, BatchNormalization, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
import os


class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        print(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][-1], input_shape[0][-1]),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)

    def call(self, x, **kwargs):
        assert isinstance(x, list)
        a, b = x
        res = tf.matmul(K.dot(a, self.kernel), tf.transpose(b, [0, 2, 1]))
        return res

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a[1], self.output_dim


def fetch_embedding_data(path, length=None):
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(',')
            temp = list(map(float, fields))
            if length and len(temp) != length:
                raise ValueError("The length of each row of data must be {}".format(length))
            data.append(temp)
    if len(data) != 166:
        raise ValueError("Samples loss")
    data_np = np.asarray(data)
    return data_np


def fetch_features(path_1, path_2, path_3, num=-1):
    print("Fetching embedding data...")
    embedding_1 = fetch_embedding_data(path_1)
    embedding_2 = fetch_embedding_data(path_2)
    embeddings = np.concatenate((embedding_1, embedding_2), axis=1)

    print("Fetching original features...")
    data = []
    label = []
    ori_emb = []
    ot, trip_num, pattern = [], [], []

    with open(path_3, 'r') as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        for line in lines[num]:
            fields = line.split(":")
            ori_station = int(fields[0])
            features = fields[1]
            target = int(fields[2])
            ot.append(int(fields[3]))
            trip_num.append(int(fields[4]))
            pattern.append(int(fields[5]))

            feature = features.split('#')
            if len(feature) != 5:
                raise ValueError("Feature loss")
            f = [list(map(float, feature[i].split(','))) for i in range(5)]
            one_line = list(zip(*f))

            ori_emb.append(embeddings[ori_station])
            data.append(one_line)
            label.append(target)

    sample_num = len(data)
    emb_shape = embeddings.shape

    embeddings_samples = np.repeat(np.reshape(embeddings, (1, emb_shape[0], emb_shape[1])), sample_num, axis=0)
    ori_emb_samples = np.asarray(ori_emb).reshape((sample_num, 1, len(embeddings[0])))
    original_features = np.asarray(data)
    labels = np.asarray(label)

    print("Fetching finished, total %d samples" % len(labels))
    return embeddings_samples, ori_emb_samples, original_features, labels, ot, trip_num, pattern


def main():
    # set GPU memory
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    path_1 = "../data/embedding_line"
    path_2 = "../data/embedding_lda"
    path_3 = "../data/features"
    embeddings_samples, ori_emb_samples, original_features, labels, ot, trips, pattern = fetch_features(path_1, path_2, path_3, -1)
    labels_one_hot = to_categorical(labels, num_classes=166)

    split = int(len(labels) * 0.9)

    train_X = [embeddings_samples[:split], ori_emb_samples[:split], original_features[:split]]
    train_y = labels_one_hot[:split]

    test_X = [embeddings_samples[split:], ori_emb_samples[split:], original_features[split:]]
    test_y = labels_one_hot[split:]
    ot_test = ot[split:]

    input_1 = Input(shape=embeddings_samples.shape[1:])
    input_2 = Input(shape=ori_emb_samples.shape[1:])
    layer_1 = MyLayer(1)([input_1, input_2])
    input_3 = Input(shape=original_features.shape[1:])

    final_feature = layers.concatenate([layer_1, input_3])
    bn_final_feature = BatchNormalization(axis=1)(final_feature)

    dense1 = Dense(128, activation='relu')(bn_final_feature)
    dense2 = Dense(64, activation='relu')(dense1)
    dense3 = Dense(64, activation='relu')(dense2)

    conv1 = Conv1D(filters=128, kernel_size=1, activation='relu')(dense3)
    bn_conv1 = BatchNormalization()(conv1)
    # dense1 = Dense(128, activation='relu')(bn_conv1)

    conv2 = Conv1D(filters=64, kernel_size=1, activation='relu')(bn_conv1)
    bn_conv2 = BatchNormalization()(conv2)
    # dense2 = Dense(64, activation='relu')(bn_conv2)

    conv3 = Conv1D(filters=64, kernel_size=1, activation='relu')(bn_conv2)
    # dense3 = Dense(32, activation='relu')(conv3)
    # conv4 = Conv1D(filters=20, kernel_size=1, activation='relu')(conv3)
    # conv5 = Conv1D(filters=20, kernel_size=1, activation='relu')(conv4)

    conv6 = Conv1D(filters=1, kernel_size=1, activation='relu')(conv3)

    flaten_conv_output = tf.reshape(conv6, shape=(-1, 166))
    # dense_1 = Dense(64, activation='relu')(flaten_conv_output)
    # dense_2 = Dense(64, activation='relu')(dense_1)
    # dense_3 = Dense(166, activation='relu')(dense_2)

    output = Softmax()(flaten_conv_output)
    model = Model(inputs=[input_1, input_2, input_3], outputs=output)
    model.summary()

    my_optimizer = optimizers.Adam(learning_rate=1e-5)
    my_loss = losses.CategoricalCrossentropy()

    model.compile(optimizer=my_optimizer,
                  loss=my_loss,
                  metrics=[metrics.categorical_accuracy])

    history = model.fit(x=train_X,
                        y=train_y,
                        validation_split=0.2,
                        shuffle=True,
                        batch_size=512,
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

    pred = model.predict(test_X)
    pred_argmax = np.argmax(pred, axis=1)
    true_argmax = np.argmax(test_y, axis=1)
    count = sum([1 if pred_argmax[i] == true_argmax[i] else 0 for i in range(len(pred))])
    print("\nTest %d samples, accuracy: %.2f%%" % (len(pred), count / len(pred) * 100))

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
