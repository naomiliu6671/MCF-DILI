import sys
import os
import numpy as np
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
name = 'tox21' # tox21 DILIconcern
f = open(os.path.join(BASE_DIR, '{}/raw/{}.csv'.format(name, name)), 'r').readlines()[1:]
np.random.shuffle(f)


if __name__ == "__main__":
    tasks = {}
    idx_neg = {}
    idx_pos = {}

    for index, line in enumerate(f):
        line = line.strip()
        l = line.split(",")
        size = len(l)
        if size < 1:
            continue
        smi = l[0]
        if name == 'tox21' or name == 'DILIconcern_support' or name == 'DILIconcern_query':
            for i in range(1, size):
                cur_item = l[i].strip()
                if i not in tasks:
                    tasks[i] = [[], []]
                if cur_item == "0.0" or cur_item == "0" or cur_item == 0:
                    tasks[i][0].append(smi)
                elif cur_item == "1.0" or cur_item == "1" or cur_item == 1:
                    tasks[i][1].append(smi)
    cnt_tasks = []
    for i in tasks:
        root = name + "/new/" + str(i)
        os.makedirs(root, exist_ok=True)
        os.makedirs(root + "/raw", exist_ok=True)
        os.makedirs(root + "/processed", exist_ok=True)
        file = open(root + "/raw/" + name + ".json", "w")
        file.write(json.dumps(tasks[i]))
        file.close()
        print('task:', i, len(tasks[i][0]), len(tasks[i][1]))
        cnt_tasks.append([len(tasks[i][0]), len(tasks[i][1])])
    print(cnt_tasks)
    
