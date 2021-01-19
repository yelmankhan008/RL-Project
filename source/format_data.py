"""
Total lines in data.csv = 62412220
Number of episodes = 1000000
Largest side of episode = 711
Size of data_new.csv = 61412219


File takes ~6-7 minutes to run on my machine
"""

import time
import numpy as np

start_time = time.time()
# Read Data
f = open("data.csv", "r")
data = f.readlines()
f.close()

data_new = []
history = []

for i in range(len(data)):
    if len(data[i].strip().split(',')) != 1:
        # Save episodes
        data_new.append(data[i])
    else:
        # Save episodes sizes
        history.append(data[i])

history = history[1:]
history = list(map(int, history))
# np.save("history.npy", np.array(history))
size_H = len(history)
max_episode_size = max(history)

data_clean = np.zeros((len(data_new), 4))

# 1
data_clean[:10000000] = np.array(
    list(map(lambda x: list(map(float, x.strip().split(','))), data_new[:10000000])))
# 2
data_clean[10000000:20000000] = (list(map(lambda x: list(
    map(float, x.strip().split(','))), data_new[10000000:20000000])))
# 3
data_clean[20000000:30000000] = (list(map(lambda x: list(
    map(float, x.strip().split(','))), data_new[20000000:30000000])))
# 4
data_clean[30000000:40000000] = (list(map(lambda x: list(
    map(float, x.strip().split(','))), data_new[30000000:40000000])))
# 5
data_clean[40000000:50000000] = (list(map(lambda x: list(
    map(float, x.strip().split(','))), data_new[40000000:50000000])))
# 6
data_clean[50000000:60000000] = (
    list(map(lambda x: list(map(float, x.strip().split(','))), data_new[50000000:60000000])))
# 7
data_clean[60000000:] = (
    list(map(lambda x: list(map(float, x.strip().split(','))), data_new[60000000:])))

episodic_data = np.empty(1000000, dtype=object)
index = 0
for i in range(len(history)):
    episodic_data[i] = data_clean[index: index + history[i]].transpose()
    index += history[i]
np.save("data.npy", episodic_data)

print(f'Done! Time taken: {time.time() - start_time} seconds!!!')
