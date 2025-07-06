import numpy as np

def compute_cost(x, y, w, b):
    """
        Basit doğrusal regresyon için cost (kayıp) fonksiyonunu hesaplar.

        Args:
            x (ndarray): Girdi verisi, örn: ev büyüklüğü
            y (ndarray): Hedef değer, örn: ev fiyatı
            w (float): Ağırlık (eğim)
            b (float): Bias (y-kesişim)

        Returns:
            float: Ortalama karesel hata (MSE) cost değeri
        """

    #number of training examples
    m = x.shape[0]

    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost