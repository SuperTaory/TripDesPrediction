from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.utils import to_categorical
from baseline.utils import fetch_data
from sklearn.model_selection import train_test_split
import numpy as np

X, y, ot = fetch_data("../data/features")
print(X.shape, y.shape)
y = to_categorical(y, num_classes=166)

train_X, test_X, train_y, test_y, _, ot_test = train_test_split(X, y, ot, test_size=0.2)
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train_X, train_y)
print("Samples: %d, Accuracy: %.2f%%" % (len(test_X), rfc.score(test_X, test_y)))

pred = rfc.predict(test_X)
pred_argmax = np.argmax(pred, axis=1)
true_argmax = np.argmax(test_y, axis=1)

ot_acc, ot_num = [0] * 12, [0] * 12
for i in range(len(pred)):
    ot_num[ot_test[i]] += 1
    if pred_argmax[i] == true_argmax[i]:
        ot_acc[ot_test[i]] += 1
ot_acc = [round(ot_acc[i] / ot_num[i], 4) if ot_num[i] else 0 for i in range(len(ot_num))]
print("ot-acc distribution")
print(ot_acc)

