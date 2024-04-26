import numpy as np

x = np.array([[1, 1, 1], [1, 1, 1]])
y = np.array([[2, 2, 2], [2, 2, 2]])

s = np.concatenate(x,y,axis=1)
print(s)