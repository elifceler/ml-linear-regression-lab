import numpy as np

def compute_model_output(x, w, b):
    """
              Basit doğrusal regresyon çıktısını hesaplar.
              Args:
                  x (np.ndarray): Girdi verisi
                  w (float): Ağırlık (slope)
                  b (float): Bias (y-intercept)
              Returns:
                  np.ndarray: Her bir x için modelin tahmini çıktısı
              """

    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb
