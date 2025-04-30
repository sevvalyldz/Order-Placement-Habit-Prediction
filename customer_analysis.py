import pandas as pd
import psycopg2
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Veritabanı bağlantı parametreleri
conn_params = {
    'dbname': 'GYKNorthwind',
    'user': 'postgres',
    'password': '47164716',
    'host': 'localhost',
    'port': '5432'
}

# Müşteri analizi için SQL sorgusu
customer_query = """
WITH monthly_orders AS (
    SELECT 
        EXTRACT(MONTH FROM o.order_date) as month,
        COUNT(DISTINCT o.order_id) as order_count,
        SUM(od.unit_price * od.quantity * (1 - od.discount)) as total_revenue
    FROM orders o
    JOIN order_details od ON o.order_id = od.order_id
    GROUP BY EXTRACT(MONTH FROM o.order_date)
)
SELECT 
    month,
    order_count,
    total_revenue
FROM monthly_orders
ORDER BY month;
"""

try:
    # Veritabanına bağlan
    conn = psycopg2.connect(**conn_params)
    conn.set_client_encoding('LATIN5')
    
    # Sorguyu çalıştır
    df = pd.read_sql_query(customer_query, conn)
    
    # Ay isimlerini ekle
    month_names = {
        1: 'Ocak', 2: 'Şubat', 3: 'Mart', 4: 'Nisan',
        5: 'Mayıs', 6: 'Haziran', 7: 'Temmuz', 8: 'Ağustos',
        9: 'Eylül', 10: 'Ekim', 11: 'Kasım', 12: 'Aralık'
    }
    df['month_name'] = df['month'].map(month_names)
    
    # Görselleştirme
    plt.figure(figsize=(15, 10))
    
    # Sipariş sayısı grafiği
    plt.subplot(2, 1, 1)
    sns.barplot(x='month_name', y='order_count', data=df)
    plt.title('Aylık Sipariş Sayısı')
    plt.xlabel('Ay')
    plt.ylabel('Sipariş Sayısı')
    plt.xticks(rotation=45)
    
    # Gelir grafiği
    plt.subplot(2, 1, 2)
    sns.barplot(x='month_name', y='total_revenue', data=df)
    plt.title('Aylık Toplam Gelir')
    plt.xlabel('Ay')
    plt.ylabel('Toplam Gelir')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data/seasonality_analysis.png')
    print("\nMevsimsellik analizi 'data/seasonality_analysis.png' dosyasına kaydedildi.")
    
    # Mevsimsellik özeti
    print("\nMevsimsellik Analizi Özeti:")
    print("\nEn çok sipariş alınan aylar:")
    top_months = df.nlargest(3, 'order_count')
    for _, row in top_months.iterrows():
        print(f"{row['month_name']}: {row['order_count']} sipariş")
    
    print("\nEn yüksek gelir elde edilen aylar:")
    top_revenue = df.nlargest(3, 'total_revenue')
    for _, row in top_revenue.iterrows():
        print(f"{row['month_name']}: {row['total_revenue']:.2f} gelir")
    
    # CSV dosyasına kaydet
    df.to_csv('data/monthly_analysis.csv', index=False)
    print("\nAylık analiz verileri 'data/monthly_analysis.csv' dosyasına kaydedildi.")
    
except Exception as e:
    print(f"Hata oluştu: {str(e)}")
finally:
    if 'conn' in locals():
        conn.close()
