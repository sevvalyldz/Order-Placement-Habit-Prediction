import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from train_model import CustomerModel
import os

def predict_customers():
    # Veriyi oku
    df = pd.read_csv('data/customer_analysis.csv')
    
    # Özellikleri seç
    features = ['total_orders', 'total_spent', 'average_order_value', 'days_since_last_order']
    X = df[features].values
    
    # Modeli yükle
    model = CustomerModel(input_size=len(features))
    model.load_state_dict(torch.load('models/customer_model.pth'))
    model.eval()
    
    # Tahminleri yap
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        predictions = model(X_tensor).numpy()
    
    # Sonuçları DataFrame'e ekle
    df['prediction_probability'] = predictions
    df['predicted_will_order'] = (predictions > 0.5).astype(int)
    
    # Sonuçları analiz et
    print("\nTahmin Sonuçları:")
    print(f"Toplam müşteri sayısı: {len(df)}")
    print(f"Tahmin edilen sipariş verecek müşteri sayısı: {df['predicted_will_order'].sum()}")
    print(f"Tahmin edilen sipariş vermeyecek müşteri sayısı: {len(df) - df['predicted_will_order'].sum()}")
    
    # En yüksek olasılıklı 5 müşteriyi göster
    print("\nEn yüksek olasılıklı 5 müşteri:")
    top_customers = df.nlargest(5, 'prediction_probability')
    print(top_customers[['customer_id', 'company_name', 'prediction_probability', 'predicted_will_order']])
    
    # Sonuçları kaydet
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/customer_predictions.csv', index=False)
    print("\nTahminler 'data/customer_predictions.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    predict_customers() 