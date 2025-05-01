from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
from typing import List
import uvicorn
import logging

# Logging ayarları
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# FastAPI uygulamasını oluştur
app = FastAPI(
    title="Müşteri Tahmin API",
    description="Müşteri davranışlarını tahmin eden API",
    version="1.0.0",
    debug=True
)

# Veri modelleri
class MusteriOzellikleri(BaseModel):
    total_orders: float
    total_spent: float
    average_order_value: float
    days_since_last_order: float

class TahminSonucu(BaseModel):
    tahmin_olasiligi: float
    siparis_verme_tahmini: int

class MusteriTahminSonucu(BaseModel):
    musteri_id: str
    sirket_adi: str
    tahmin_olasiligi: float
    siparis_verme_tahmini: int

# Ana sayfa endpoint'i
@app.get("/")
async def ana_sayfa():
    logger.info("Ana sayfa endpoint'i çağrıldı")
    return {"mesaj": "Müşteri Tahmin API'sine Hoş Geldiniz"}

# Tek müşteri için tahmin endpoint'i
@app.post("/tahmin", response_model=TahminSonucu)
async def musteri_tahmini(ozellikler: MusteriOzellikleri):
    try:
        logger.info(f"Tahmin endpoint'i çağrıldı. Gelen veriler: {ozellikler}")
        
        # Özellikleri listeye dönüştür
        ozellik_listesi = [
            ozellikler.total_orders,
            ozellikler.total_spent,
            ozellikler.average_order_value,
            ozellikler.days_since_last_order
        ]
        logger.debug(f"Özellik listesi oluşturuldu: {ozellik_listesi}")
        
        # Modeli yükle
        try:
            with open('models/customer_model.pkl', 'rb') as f:
                model = pickle.load(f)
                logger.info("Model başarıyla yüklendi")
        except FileNotFoundError:
            logger.error("Model dosyası bulunamadı")
            raise HTTPException(status_code=500, detail="Model dosyası bulunamadı. Lütfen önce modeli eğitin.")
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model yüklenirken hata: {str(e)}")
        
        # Tahmin yap
        try:
            tahmin = model.predict_proba([ozellik_listesi])[0][1]
            logger.info(f"Tahmin yapıldı: {tahmin}")
        except Exception as e:
            logger.error(f"Tahmin yapılırken hata: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tahmin yapılırken hata: {str(e)}")
        
        sonuc = TahminSonucu(
            tahmin_olasiligi=float(tahmin),
            siparis_verme_tahmini=int(tahmin > 0.5)
        )
        logger.info(f"Sonuç hazırlandı: {sonuc}")
        return sonuc
        
    except Exception as e:
        logger.error(f"Genel hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Tüm müşteriler için tahmin endpoint'i
@app.get("/tum-tahminler", response_model=List[MusteriTahminSonucu])
async def tum_musteri_tahminleri():
    try:
        logger.info("Tüm tahminler endpoint'i çağrıldı")
        
        # Veriyi oku
        try:
            df = pd.read_csv('data/customer_analysis.csv')
            logger.info(f"Veri dosyası okundu. Satır sayısı: {len(df)}")
        except FileNotFoundError:
            logger.error("Veri dosyası bulunamadı")
            raise HTTPException(status_code=500, detail="Veri dosyası bulunamadı")
        except Exception as e:
            logger.error(f"Veri okuma hatası: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Veri okuma hatası: {str(e)}")
        
        # Özellikleri seç
        ozellikler = ['total_orders', 'total_spent', 'average_order_value', 'days_since_last_order']
        X = df[ozellikler].values
        logger.debug(f"Özellikler seçildi. Şekil: {X.shape}")
        
        # Modeli yükle
        try:
            with open('models/customer_model.pkl', 'rb') as f:
                model = pickle.load(f)
                logger.info("Model başarıyla yüklendi")
        except FileNotFoundError:
            logger.error("Model dosyası bulunamadı")
            raise HTTPException(status_code=500, detail="Model dosyası bulunamadı. Lütfen önce modeli eğitin.")
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model yüklenirken hata: {str(e)}")
        
        # Tahminleri yap
        try:
            tahminler = model.predict_proba(X)[:, 1]
            logger.info(f"Tahminler yapıldı. Tahmin sayısı: {len(tahminler)}")
        except Exception as e:
            logger.error(f"Tahminler yapılırken hata: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Tahminler yapılırken hata: {str(e)}")
        
        # Sonuçları hazırla
        sonuclar = []
        for i, row in df.iterrows():
            sonuc = MusteriTahminSonucu(
                musteri_id=str(row['customer_id']),
                sirket_adi=str(row['company_name']),
                tahmin_olasiligi=float(tahminler[i]),
                siparis_verme_tahmini=int(tahminler[i] > 0.5)
            )
            sonuclar.append(sonuc)
        
        logger.info(f"Sonuçlar hazırlandı. Sonuç sayısı: {len(sonuclar)}")
        return sonuclar
        
    except Exception as e:
        logger.error(f"Genel hata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# API'yi çalıştır
if __name__ == "__main__":
    logger.info("API başlatılıyor...")
    uvicorn.run(
        "api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="debug"
    ) 