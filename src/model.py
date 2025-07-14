import numpy as np

def compute_model_output(x, w, b):
    """
    Çok değişkenli doğrusal regresyon çıktısını hesaplar.
    Args:
        x (np.ndarray): Girdi verisi (m, n)
        w (np.ndarray): Ağırlıklar (n,)
        b (float): Bias
    Returns:
        np.ndarray: Her bir x için modelin tahmini çıktısı
    """
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = np.dot(x[i], w) + b

    return f_wb
