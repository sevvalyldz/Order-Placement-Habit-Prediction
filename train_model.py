import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import RandomOverSampler
import os

# Veri seti sınıfı
class CustomerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Focal Loss implementasyonu
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

# Model mimarisi
class CustomerModel(nn.Module):
    def __init__(self, input_size):
        super(CustomerModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.layer4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.layer4(x))
        return x

def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch)
            test_loss += criterion(y_pred, y_batch).item()
            predicted = (y_pred > 0.5).float()
            all_predictions.extend(predicted.numpy())
            all_targets.extend(y_batch.numpy())
    
    # Metrikleri hesapla
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions)
    recall = recall_score(all_targets, all_predictions)
    f1 = f1_score(all_targets, all_predictions)
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    return test_loss/len(test_loader), accuracy, precision, recall, f1, conf_matrix

def calculate_class_weights(y):
    """Sınıf ağırlıklarını hesapla"""
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights)

def train_model():
    # Veriyi oku
    df = pd.read_csv('data/enhanced_customer_analysis.csv')
    
    # Özellikleri ve hedef değişkeni seç
    features = ['total_orders', 'total_spent', 'average_order_value', 'days_since_last_order']
    X = df[features].values
    y = df['will_order_next_6months'].values
    
    # Sınıf dağılımını kontrol et
    class_distribution = pd.Series(y).value_counts()
    print("\nOrijinal Sınıf Dağılımı:")
    print(class_distribution)
    print(f"Azınlık sınıf oranı: {min(class_distribution) / len(y):.2%}")
    
    # Veriyi eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Random oversampling uygula
    ros = RandomOverSampler(random_state=42, sampling_strategy=0.5)  # Azınlık sınıfını çoğunluk sınıfının %50'si kadar yap
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)
    
    print("\nOversampling Sonrası Eğitim Seti Sınıf Dağılımı:")
    print(pd.Series(y_train_resampled).value_counts())
    
    # Veri setlerini oluştur
    train_dataset = CustomerDataset(X_train_resampled, y_train_resampled)
    test_dataset = CustomerDataset(X_test, y_test)
    
    # DataLoader'ları oluştur
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model, kayıp fonksiyonu ve optimizer tanımla
    model = CustomerModel(input_size=len(features))
    criterion = FocalLoss(alpha=0.25, gamma=2)  # Focal Loss kullan
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Eğitim döngüsü
    num_epochs = 100
    best_test_loss = float('inf')
    best_metrics = None
    
    print("\nModel Eğitimi Başladı:")
    print("------------------------")
    
    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Test ve değerlendirme
        test_loss, accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, test_loader, criterion)
        
        # Her 10 epoch'ta bir sonuçları yazdır
        if (epoch + 1) % 10 == 0:
            print(f'\nEpoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}')
            print(f'Test Loss: {test_loss:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print('\nConfusion Matrix:')
            print(conf_matrix)
            print('------------------------')
        
        # En iyi modeli kaydet
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix
            }
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/customer_model.pth')
    
    print("\nEğitim tamamlandı!")
    print("\nEn İyi Model Metrikleri:")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print("\nConfusion Matrix:")
    print(best_metrics['confusion_matrix'])
    print("\nModel 'models/customer_model.pth' dosyasına kaydedildi.")

if __name__ == "__main__":
    train_model() 