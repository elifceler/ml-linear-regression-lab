# 🧮 Multiple Linear Regression from Scratch (California Housing)

Bu proje, sıfırdan çok değişkenli lineer regresyon modelinin nasıl geliştirileceğini gösteren sade ama güçlü bir örnektir. Amacım yalnızca `sklearn` gibi kütüphaneleri kullanmak değil, **regresyonun matematiksel mantığını ve optimizasyon sürecini** manuel olarak uygulamaktır.

---

## 🔍 Problem Tanımı

California Housing veri seti kullanılarak bir evin medyan fiyatı, 8 farklı özellik üzerinden tahmin edilmeye çalışıldı:

- MedInc (Gelir)
- HouseAge (Ev Yaşı)
- AveRooms (Ortalama Oda Sayısı)
- AveBedrms (Ortalama Yatak Odası)
- Population (Nüfus)
- AveOccup (Ortalama Kişi Sayısı)
- Latitude (Enlem)
- Longitude (Boylam)

---

## ⚙️ Kullanılan Yöntemler

- **Gradient Descent (Manuel)**: Ağırlık ve bias değerlerini iteratif olarak optimize ettik.
- **MSE (Mean Squared Error)**: Kayıp fonksiyonu olarak kullanıldı.
- **Veri Normalizasyonu**: Özelliklerin dağılımı normalize edildi.
- **Model Kaydetme**: Eğitilen ağırlıklar `.npy` formatında saklandı.
- **Sklearn Karşılaştırması**: `LinearRegression` ile performans farkı analiz edildi.

---

## 🧪 Sonuçlar

| Metrik             | Kendi Modelim | `sklearn.LinearRegression` |
|--------------------|---------------|-----------------------------|
| R² Skoru           | 0.5672        | 0.5758                      |
| MAE (Ortalama Hata)| 0.55          | 0.53                        |

📉 Ayrıca `cost` fonksiyonunun zamanla nasıl azaldığı da görselleştirildi.

---

## 📊 Özellik Önem Grafiği

Aşağıda kendi modelimin hesapladığı ağırlık değerleri yer alıyor:

```python
plt.barh(feature_names, w)
plt.title("Özellik Önem Grafiği")
```

Bu grafik hangi değişkenlerin ev fiyatları üzerinde ne kadar etkili olduğunu anlamak için kullanıldı. Örneğin `MedInc` ve `HouseAge` en belirgin pozitif etkilerden biri oldu.

---

## 🗂️ Proje Yapısı

```
├── main.py               # Eğitim süreci
├── analyze.py            # Modeli test etme ve görselleştirme
├── california_housing.py # Veri setini yükleme
├── gradient_descent.py   # Gradient Descent algoritması
├── cost_function.py      # Kayıp fonksiyonu (MSE)
├── model.py              # Tahmin fonksiyonu (w*x + b)
├── plots/                # Grafik çıktıları
├── README.md             # Bu dosya
```

---

## 🚀 Nasıl Çalıştırılır?

```bash
pip install -r requirements.txt
python main.py         # Modeli eğitir
python analyze.py      # Sonuçları analiz eder
```

---

## 🎯 Neden Bu Proje?

Bu projeyi yalnızca bir ödev olarak değil, **model öğrenimini içselleştirmek**, **“hazır model kullanmadan öğrenmek”** ve gerçek veriye karşı nasıl performans gösterdiğini test etmek amacıyla yaptım.

---

## 🧠 Kazanımlarım

- Gradient descent mantığını derinlemesine kavradım.
- Model ve metrik karşılaştırması yapmayı öğrendim.
- Kodun her satırının ne yaptığını anlayarak yazdım.

---

## 📌 Not

Bu proje, yapay zekâ alanında sağlam bir temel oluşturmak isteyen herkes için önerilir. "Model nasıl çalışıyor?" sorusuna cevap arayanlar için sade ama öğretici bir örnektir.

---

## 📬 İletişim

Her türlü öneri, geri bildirim veya işbirliği için ulaşabilirsiniz:  
📧 elifceler55@gmail.com  
📍 GitHub: [github.com/elifceler](https://github.com/elifceler)
