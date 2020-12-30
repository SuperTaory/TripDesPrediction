from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from baseline.utils import fetch_data

X, y, ot = fetch_data("../data/features")
print(X.shape, y.shape)

X_train, X_test, y_train, y_test, _, ot_test = train_test_split(X, y, ot, test_size=0.2, shuffle=True)
print(X_train.shape, y_train.shape)

model = XGBClassifier(objective="multi:softprob")
model.fit(X_train, y_train)
print("Samples: %d, Accuracy: %.2f%%" % (len(X_test), model.score(X_test, y_test)))

pred = model.predict(X_test)
ot_num, ot_acc = [0] * 12, [0] * 12
for i in range(len(pred)):
    ot_num[ot_test[i]] += 1
    if pred[i] == y_test[i]:
        ot_acc[ot_test[i]] += 1
ot_acc = [round(ot_acc[i] / ot_num[i], 4) if ot_num[i] else 0 for i in range(len(ot_num))]
print("ot-acc distribution")
print(ot_acc)
