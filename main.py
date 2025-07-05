from data import x_train, y_train
from model import compute_model_output
import matplotlib.pyplot as plt

# Model parametreleri
w = 200
b = 100

# Modelin tahmini
f_wb = compute_model_output(x_train, w, b)

#Veri noktalarını çiz
plt.scatter(x_train, y_train, marker='x', color='red', label='Gerçek Değerler') #scatter noktaları bağımsız işaretler olarak gösterir

# Model tahmin çizgisi
plt.plot(x_train, f_wb, color='blue', label='Model Tahmini') # çizgi çizer, marker eklersen nokta da gösterir.

#Grafik ayarları
plt.title("Housing Prices")
plt.xlabel("Ev Boyutu (1000 sqft)") # x ekseni etiketi
plt.ylabel("Fiyat (1000$)") # y ekseni etiketi
plt.legend() # etiket kutusunu göster
plt.grid(True) # ızgara çizgileri
plt.show() # hepsini ekrana bas

# Tahmin yapmak istediğimiz evin büyüklüğü (1000 sqft cinsinden)
x_input = 1.2

#Modelin tahmini (vektörleşmeye gerek yok, tek sayı)
predicted_price = w * x_input + b

#Sonucu yazdır
print(f"{x_input*1000:.0f} sqft'lik evin tahmini fiyatı: ${predicted_price:.0f}K")