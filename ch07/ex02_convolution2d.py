"""
2차원 Convolution(합성곱) 연산
"""
import numpy as np


def convolution_2d(x, w):
    """x, w: 2d ndarray. x.shape >= w.shape
    x와 w의 교차 상관 연산 결과를 리턴
    """
    # convolution 결과 행렬(2d ndarray)의 shape: (rows, cols)
    rows = x.shape[0] - w.shape[0] + 1
    cols = x.shape[1] - w.shape[1] + 1
    conv = []  # 결과를 저장할 리스트
    for i in range(rows):
        for j in range(cols):
            x_sub = x[i:(i + w.shape[0]), j:(j + w.shape[1])]
            fma = np.sum(x_sub * w)
            conv.append(fma)
    conv = np.array(conv)
    return conv.reshape((rows, cols))


if __name__ == '__main__':
    np.random.seed(113)

    x = np.arange(1, 10).reshape((3, 3))
    print(x)
    w = np.array([[2, 0],
                  [0, 0]])
    print(w)

    # 2d 배열 x의 가로(width) xw, 세로(hight) xh
    xh, xw = x.shape[0], x.shape[1]
    # 2d 배열 w의 가로 ww, 세로 wh
    wh, ww = w.shape[0], w.shape[1]

    x_sub1 = x[0:wh, 0:ww]  # x[0:2, 0:2]
    print(x_sub1)
    fma1 = np.sum(x_sub1 * w)
    print(fma1)
    x_sub2 = x[0:wh, 1:1+ww]  # x[0:2, 1:3]
    print(x_sub2)
    fma2 = np.sum(x_sub2 * w)
    x_sub3 = x[1:1+wh, 0:ww]  # x[1:3, 0:2]
    fma3 = np.sum(x_sub3 * w)
    x_sub4 = x[1:1+wh, 1:1+ww]  # x[1:3, 1:3]
    fma4 = np.sum(x_sub4 * w)
    conv = np.array([fma1, fma2, fma3, fma4]).reshape((2, 2))
    print(conv)

    conv = convolution_2d(x, w)
    print(conv)

    x = np.random.randint(10, size=(5, 5))
    w = np.random.randint(5, size=(3, 3))
    print(x)
    print(w)
    print(convolution_2d(x, w))





