import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load data
df = pd.read_csv('data/customer_analysis.csv')

# Prepare features and target
features = ['total_orders', 'total_spent', 'average_order_value', 'days_since_last_order']
X = df[features]
y = df['will_order_next_6months']  # Updated target column name

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Train accuracy: {train_score:.3f}")
print(f"Test accuracy: {test_score:.3f}")

# Save model
with open('models/customer_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model saved successfully!") 