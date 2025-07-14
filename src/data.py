import numpy as np

# Girdi verisi: [ev büyüklüğü (1000 sqft), oda sayısı]
x_train = np.array([
    [1.0, 2],
    [1.7, 3],
    [2.0, 3],
    [2.5, 4],
    [3.0, 4],
    [3.2, 5]
])

# Hedef değerler: fiyat (1000$ cinsinden)
y_train = np.array([250, 300, 480, 430, 630, 730])
