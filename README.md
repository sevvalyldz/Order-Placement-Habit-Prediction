# Müşteri Sipariş Tahmin Projesi

Bu proje, müşterilerin gelecek 6 ay içinde sipariş verip vermeyeceğini tahmin etmek için derin öğrenme tabanlı bir model geliştirmeyi amaçlamaktadır.

## Proje Yapısı

```
.
├── data/
│   ├── customer_analysis.csv
│   └── enhanced_customer_analysis.csv
├── models/
│   └── customer_model.pth
├── src/
│   ├── customer_analysis.py
│   └── customer_prediction.py
├── train_model.py
└── README.md
```

## Gereksinimler

Projeyi çalıştırmak için aşağıdaki Python kütüphanelerine ihtiyaç vardır:

```bash
pip install pandas numpy torch scikit-learn imbalanced-learn
```

## Özellikler

- Müşteri sipariş verilerinin analizi
- Derin öğrenme tabanlı tahmin modeli
- Dengesiz veri seti yönetimi
- Model performans değerlendirmesi

## Kullanım

1. Veri Analizi:
```bash
python src/customer_analysis.py
```

2. Model Eğitimi:
```bash
python train_model.py
```

3. Tahmin Yapma:
```bash
python src/customer_prediction.py
```

## Model Detayları

### Mimari
- 4 katmanlı sinir ağı
- Giriş katmanı: 4 özellik
- Gizli katmanlar: 64 -> 32 -> 16 nöron
- Çıkış katmanı: 1 nöron (sigmoid aktivasyon)

### Özellikler
- total_orders: Toplam sipariş sayısı
- total_spent: Toplam harcama
- average_order_value: Ortalama sipariş değeri
- days_since_last_order: Son siparişten bu yana geçen gün sayısı

### Performans Metrikleri
- Accuracy: 0.9815
- Precision: 0.9815
- Recall: 1.0000
- F1 Score: 0.9907

## Veri Seti

Veri seti aşağıdaki bilgileri içermektedir:
- Müşteri sipariş geçmişi
- Sipariş değerleri
- Sipariş zamanları
- Müşteri davranış metrikleri

## Geliştirme

### Dengesiz Veri Seti Yönetimi
- Random Oversampling kullanıldı
- Focal Loss ile azınlık sınıfına önem verildi
- Sınıf ağırlıkları hesaplandı

### Model İyileştirmeleri
- Dropout (0.2) ile regularizasyon
- Adam optimizer
- Öğrenme oranı: 0.001
- Batch size: 32

## Gelecek Geliştirmeler

1. Veri Toplama
   - Daha fazla negatif örnek toplanması
   - Yeni özellikler eklenmesi

2. Model İyileştirmeleri
   - Farklı oversampling tekniklerinin denenmesi
   - Model mimarisinin basitleştirilmesi
   - Regularizasyon parametrelerinin artırılması

3. Değerlendirme
   - Cross-validation uygulanması
   - Farklı metriklerin eklenmesi
   - Model açıklanabilirliğinin artırılması

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## İletişim

Sorularınız veya önerileriniz için lütfen iletişime geçin. 