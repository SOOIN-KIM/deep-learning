# 그림 3-14(p.82)
import numpy as np

x = np.array([1, 2])
W1 = np.array([[1, 4],
              [2, 5],
              [3, 6]])
b = 1
y = W1.dot(x) + b
print(y)

W2 = np.array([[1, 2, 3],
               [4, 5, 6]])
y = x.dot(W2) + 1
print(y)
