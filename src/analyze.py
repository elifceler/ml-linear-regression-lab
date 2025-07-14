import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from california_housing import load_california_data
from model import compute_model_output
from sklearn.linear_model import LinearRegression

# Veriyi yükle
x_train, x_test, y_train, y_test = load_california_data()

# Normalizasyon
mean = np.mean(x_train, axis=0)
std = np.std(x_train, axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Eğitilmiş parametreleri yükle
w = np.load("data/trained_weights.npy")
b = np.load("data/trained_bias.npy")

# Kendi modelinle tahmin
y_pred = compute_model_output(x_test, w, b)

# Performans metrikleri
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Test Sonuçları (Kendi Modelin) ---")
print(f"R² Skoru: {r2:.4f}")
print(f"MAE: {mae:.2f}")

# sklearn karşılaştırması (normalize edilmiş verilerle!)
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred_sklearn = lr_model.predict(x_test)

r2_sklearn = r2_score(y_test, y_pred_sklearn)
mae_sklearn = mean_absolute_error(y_test, y_pred_sklearn)

print("\n--- sklearn.Model Karşılaştırması ---")
print(f"R² (sklearn): {r2_sklearn:.4f}")
print(f"MAE (sklearn): {mae_sklearn:.2f}")

# Özellik isimleri
feature_names = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude"
]

# Özellik ağırlıklarını çiz
plt.figure(figsize=(10,5))
plt.barh(feature_names, w)
plt.xlabel("Ağırlık (w)")
plt.title("Özellik Önem Grafiği (Kendi Modelin)")
plt.grid(True)
plt.tight_layout()
plt.show()
