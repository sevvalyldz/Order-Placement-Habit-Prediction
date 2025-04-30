import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_synthetic_customers(original_df, multiplier=2):
    """
    Orijinal veri setinden sentetik müşteri verileri üretir.
    
    Args:
        original_df: Orijinal müşteri verileri
        multiplier: Kaç kat daha fazla veri üretileceği
    
    Returns:
        Sentetik müşteri verileri
    """
    # Orijinal verilerin istatistiklerini hesapla
    stats = {
        'total_orders': {
            'mean': original_df['total_orders'].mean(),
            'std': original_df['total_orders'].std()
        },
        'total_spent': {
            'mean': original_df['total_spent'].mean(),
            'std': original_df['total_spent'].std()
        },
        'average_order_value': {
            'mean': original_df['average_order_value'].mean(),
            'std': original_df['average_order_value'].std()
        },
        'days_since_last_order': {
            'mean': original_df['days_since_last_order'].mean(),
            'std': original_df['days_since_last_order'].std()
        }
    }
    
    # Yeni müşteri sayısı
    n_new_customers = len(original_df) * multiplier
    
    # Sentetik veri üret
    synthetic_data = {
        'customer_id': [f'SYNTH_{i+1}' for i in range(n_new_customers)],
        'company_name': [f'Synthetic Company {i+1}' for i in range(n_new_customers)],
        'total_orders': np.random.normal(stats['total_orders']['mean'], 
                                       stats['total_orders']['std'], 
                                       n_new_customers).astype(int),
        'total_spent': np.random.normal(stats['total_spent']['mean'], 
                                      stats['total_spent']['std'], 
                                      n_new_customers),
        'average_order_value': np.random.normal(stats['average_order_value']['mean'], 
                                              stats['average_order_value']['std'], 
                                              n_new_customers),
        'days_since_last_order': np.random.normal(stats['days_since_last_order']['mean'], 
                                                stats['days_since_last_order']['std'], 
                                                n_new_customers).astype(int)
    }
    
    # Negatif değerleri düzelt
    synthetic_data['total_orders'] = np.maximum(synthetic_data['total_orders'], 0)
    synthetic_data['total_spent'] = np.maximum(synthetic_data['total_spent'], 0)
    synthetic_data['average_order_value'] = np.maximum(synthetic_data['average_order_value'], 0)
    synthetic_data['days_since_last_order'] = np.maximum(synthetic_data['days_since_last_order'], 0)
    
    # will_order_next_6months değerini hesapla
    synthetic_data['will_order_next_6months'] = (synthetic_data['days_since_last_order'] <= 180).astype(int)
    
    # DataFrame oluştur
    synthetic_df = pd.DataFrame(synthetic_data)
    
    return synthetic_df

def main():
    # Orijinal verileri oku
    original_df = pd.read_csv('data/customer_analysis.csv')
    
    # Sentetik veri üret
    synthetic_df = generate_synthetic_customers(original_df, multiplier=2)
    
    # Orijinal ve sentetik verileri birleştir
    combined_df = pd.concat([original_df, synthetic_df], ignore_index=True)
    
    # Veriyi karıştır
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Sonuçları kaydet
    combined_df.to_csv('data/enhanced_customer_analysis.csv', index=False)
    print(f"\nVeri seti genişletildi:")
    print(f"Orijinal veri sayısı: {len(original_df)}")
    print(f"Sentetik veri sayısı: {len(synthetic_df)}")
    print(f"Toplam veri sayısı: {len(combined_df)}")
    print("\nGenişletilmiş veri seti 'data/enhanced_customer_analysis.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    main() 