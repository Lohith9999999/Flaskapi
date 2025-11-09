import os
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# Output paths (same folder as your Flask app)
BASEDIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASEDIR, "model.pkl")
SCALER_PATH = os.path.join(BASEDIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASEDIR, "feature_names.pkl")

# Create synthetic regression data (replace with your CSV loading if needed)
# n_features controls how many features your model expects
n_samples = 200
n_features = 4
X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=10.0, random_state=42)

# Optionally convert to meaningful feature names
feature_names = [f"f{i+1}" for i in range(n_features)]

# Fit scaler and linear regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LinearRegression()
model.fit(X_scaled, y)

# Save artifacts
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

with open(FEATURES_PATH, "wb") as f :
    pickle.dump(feature_names, f)

print("Saved:", MODEL_PATH)
print("Saved:", SCALER_PATH)
print("Saved:", FEATURES_PATH)


