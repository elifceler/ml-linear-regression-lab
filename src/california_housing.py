from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def load_california_data():
    data = fetch_california_housing()
    X, y = data.data, data.target  # X.shape = (20640, 8), y.shape = (20640,)

    # Normalize et
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/Test split
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train, y_test
