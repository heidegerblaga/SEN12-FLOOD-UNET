import os
import rasterio
import numpy as np

# Ścieżka do folderu SEN12FLOOD
base_dir = "C:/Users/skyri/Desktop/SEN12FLOOD"

# Iteracja przez każdy podfolder
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Sprawdzanie, czy podfolder faktycznie istnieje
    if os.path.isdir(folder_path):
        
        # Wyszukiwanie plików TIF w folderze
        tif_files = [f for f in os.listdir(folder_path) if f.endswith(".tif")]
        
        for tif_file in tif_files:
            tif_path = os.path.join(folder_path, tif_file)
            
            # Wczytanie obrazu satelitarnego
            with rasterio.open(tif_path) as src:
                image_data = src.read()  # Wczytanie wszystkich pasm
                meta = src.meta  # Pobranie metadanych obrazu
                
            # Przetwarzanie obrazu - normalizacja do zakresu [0, 1]
            image_data = image_data / 255.0  # Normalizacja, zakładamy 8-bitowe obrazy
            
            # Dodatkowa obróbka, np. wyodrębnienie jednego z pasm
            # Zakładamy, że pasmo nr 1 to przykład pasma do dalszej analizy
            band1 = image_data[0, :, :]  # Pierwsze pasmo jako przykład
            
            # Można dodać kod do zapisywania przetworzonego obrazu, dalszej analizy, itp.
            print(f"Przetworzono plik: {tif_file} w folderze: {folder_name}")
            print("Wymiary obrazu:", band1.shape)
            print("Zakres wartości w pierwszym paśmie:", band1.min(), band1.max())

            # Przykładowe dalsze przetwarzanie
            # np. Zapis przetworzonego obrazu lub wyekstrahowanej maski
            # with rasterio.open("output_path", "w", **meta) as dst:
            #     dst.write(band1, 1)  # Zapis pierwszego pasma jako przykład
