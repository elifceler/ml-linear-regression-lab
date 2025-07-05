# Basit Lineer Regresyon — Konut Fiyat Tahmini 🏡📈

Bu proje, Andrew Ng'nin Machine Learning Specialization kursu kapsamında yapılmış basit bir lineer regresyon uygulamasıdır.

## 🚀 Amaç

1000 sqft ve 2000 sqft büyüklüğündeki iki evin fiyatına bakarak,
benzer büyüklükteki diğer evlerin fiyatlarını tahmin eden bir model inşa etmek.

## 🧠 Model

Model, `f_wb(x) = wx + b` şeklinde doğrusal bir denklemle tahmin yapar.
- `w`: Ağırlık (eğim)
- `b`: Bias (y-kesişim)

## 🛠 Kullanılan Kütüphaneler
- `NumPy`: Hızlı matematiksel işlemler
- `Matplotlib`: Veri görselleştirme

## 📂 Dosya Yapısı

```bash
├── model.py         # Model fonksiyonu
├── main.py          # Görselleştirme ve tahmin çıktısı
├── plots/           # Grafik çıktıları (isteğe bağlı)
└── README.md        # Bu dosya

```

## 📊 Tahmin

Modeli çalıştırarak, 1200 sqft'lik bir evin fiyatı tahmin edilebilir:  
Örnek çıktı: `1200.0 sqft'lik evin tahmini fiyatı: 340000.0$`

---

## ✨ Geliştirme Önerileri

- Maliyet fonksiyonu (cost function) ekleyip `w` ve `b`’yi otomatik öğrenme  
- Daha fazla veri ile çalışmak  
- Gradient descent uygulamak  
