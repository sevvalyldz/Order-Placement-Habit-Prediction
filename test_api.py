import requests
import json

# API'nin çalıştığı adres
BASE_URL = "http://127.0.0.1:8000"

def test_ana_sayfa():
    response = requests.get(f"{BASE_URL}/")
    print("Ana Sayfa Yanıtı:", response.json())

def test_tek_musteri_tahmini():
    # Test verisi
    test_veri = {
        "total_orders": 5,
        "total_spent": 1000,
        "average_order_value": 200,
        "days_since_last_order": 30
    }
    
    # POST isteği gönder
    response = requests.post(
        f"{BASE_URL}/tahmin",
        json=test_veri
    )
    print("\nTek Müşteri Tahmin Yanıtı:", response.json())

def test_tum_musteriler():
    response = requests.get(f"{BASE_URL}/tum-tahminler")
    print("\nTüm Müşteriler Yanıtı:", json.dumps(response.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    print("API Testleri Başlıyor...")
    test_ana_sayfa()
    test_tek_musteri_tahmini()
    test_tum_musteriler() 