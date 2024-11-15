import numpy as np
import pandas as pd
import rasterio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

# Wczytanie i przetworzenie obrazu satelitarnego
def load_data(file_path):
    with rasterio.open(file_path) as src:
        data = src.read()
    return data

# Przykładowe obliczenie cech - np. Normalized Difference Water Index (NDWI)
def calculate_features(sar_data, optical_data):
    # Przykładowo, NDWI = (Green - NIR) / (Green + NIR)
    green_band = optical_data[2, :, :]  # np. indeks pasma 2 jako zielone
    nir_band = optical_data[7, :, :]  # np. indeks pasma 7 jako bliskie podczerwone
    ndwi = (green_band - nir_band) / (green_band + nir_band + 1e-10)
    return ndwi

# Przygotowanie zestawu danych
def prepare_data(sar_file, optical_file):
    sar_data = load_data(sar_file)
    optical_data = load_data(optical_file)
    
    # Feature Engineering
    ndwi = calculate_features(sar_data, optical_data)
    
    # Normalizacja cech
    scaler = StandardScaler()
    ndwi_scaled = scaler.fit_transform(ndwi.reshape(-1, 1))
    
    # Przygotowanie końcowego zestawu danych
    X = np.column_stack([sar_data.reshape(-1), ndwi_scaled])
    y = np.random.randint(0, 2, size=X.shape[0])  # Przykładowe etykiety
    return X, y
