import numpy as np


def fetch_data(path, num=-1):
    data = []
    label = []
    ot = []
    with open(path, 'r') as f:
        lines = f.readlines()
        np.random.shuffle(lines)
        # start = int(np.random.rand() * (len(lines) - num)) if num != -1 else 0
        for line in lines[:num]:
            one_line = []
            fields = line.split(":")
            ori_station = int(fields[0])
            features = fields[1]
            target = int(fields[2])
            ot.append(int(fields[3]))

            feature = features.split('#')
            # if len(feature) != 5:
            #     raise ValueError("feature loss")
            f = [list(map(float, feature[i].split(','))) for i in range(5)]
            # if any([len(f[j]) != 166 for j in range(4)]):
            #     raise ValueError("err:166")
            # for k in range(166):
            #     one_line.append([float(f[i][k]) for i in range(5)])
            one_line = f[0] + f[1] + f[2] + f[3] + f[4]
            data.append(one_line)
            label.append(target)
    return np.asarray(data), np.asarray(label), ot
