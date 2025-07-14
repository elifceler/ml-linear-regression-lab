import numpy as np

def compute_cost(x, y, w, b):
    """
    Çoklu doğrusal regresyon için cost fonksiyonu (MSE).
    """
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = np.dot(x[i], w) + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost

    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost
