from src.california_housing import load_california_data
from src.gradient_descent import gradient_descent
from src.cost_function import compute_cost
from src.model import compute_model_output
import numpy as np
import matplotlib.pyplot as plt

# Veriyi yükle
x_train, x_test, y_train, y_test = load_california_data()

# Özellikleri standardize et
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)

x_train = (x_train - mean) / std
x_test = (x_test - mean) / std


# Başlangıç parametreleri
initial_w = np.zeros(x_train.shape[1])
initial_b = 0
learning_rate = 0.01
iterations = 1000

# Modeli eğit
w, b, cost_history = gradient_descent(x_train, y_train, initial_w, initial_b, learning_rate, iterations)

# Eğitilen parametreleri kaydet
np.save("src/trained_weights.npy", w)
np.save("src/trained_bias.npy", b)

# Cost eğrisi çiz
plt.plot(cost_history)
plt.title("Cost Zamanla Nasıl Azaldı?")
plt.xlabel("İterasyon")
plt.ylabel("Cost")
plt.grid(True)
plt.show()
