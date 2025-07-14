import numpy as np
from cost_function import compute_cost

def compute_gradient(x, y, w, b):
    """
    Çoklu regresyon için w ve b'nin gradyanını hesaplar.
    """
    m, n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0.

    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        error = f_wb - y[i]
        dj_dw += error * x[i]
        dj_db += error

    dj_dw /= m
    dj_db /= m

    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Çoklu değişkenli regresyon için gradient descent algoritması.
    """
    w = w_in.copy()
    b = b_in
    cost_history = []

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w -= alpha * dj_dw
        b -= alpha * dj_db
        cost = compute_cost(x, y, w, b)
        cost_history.append(cost)

        if i % 100 == 0:
            print(f"İterasyon {i}: Cost={cost:.2f}, w={w}, b={b:.2f}")

    return w, b, cost_history
