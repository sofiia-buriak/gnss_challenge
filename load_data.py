import pandas as pd
import os
import gc  # Garbage Collector для очищення пам'яті

def load_all_data(base_path='data/raw'):
    print("🚀 Починаємо завантаження даних...")
    files = [
        '2025-08-01_11-23-55.csv',
        '2025-10-29_07-11-02.csv',
        '2025-11-24_06-14-47.csv',
        '2025-12-11_11-06-32.csv'
    ]
    dtypes = {
        'fixType': 'int8',
        'gnssFixOk': 'int8',
        'numSV': 'int8',
        'numSats Tracked': 'int16',
        'hAcc': 'float32', 'vAcc': 'float32', 'tAcc': 'float32', 'sAcc': 'float32',
        'PDOP': 'float32', 'hDOP': 'float32', 'vDOP': 'float32', 
        'nDOP': 'float32', 'eDOP': 'float32', 'tDOP': 'float32', 'gDOP': 'float32',
        'cnoMean': 'float32', 'cnoStd': 'float32',
        'cnoMin': 'float32', 'cnoMax': 'float32',
        'overallPositionLabel': 'int8',
        'horizontalPositionLabel': 'int8',
        'verticalPositionLabel': 'int8'
    }
    dataframes = []
    for file_name in files:
        full_path = os.path.join(base_path, file_name)
        if os.path.exists(full_path):
            print(f"   ⏳ Читаю файл: {file_name}...")
            df_chunk = pd.read_csv(full_path, dtype=dtypes)
            df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], format='mixed')
            dataframes.append(df_chunk)
            print(f"      ✅ Завантажено {len(df_chunk)} рядків.")
        else:
            print(f"   ❌ Файл не знайдено: {file_name}")
    print("🔗 Об'єдную всі файли в один DataFrame...")
    full_df = pd.concat(dataframes, ignore_index=True)
    full_df = full_df.sort_values('timestamp').reset_index(drop=True)
    del dataframes
    gc.collect()
    print(f"🎉 Готово! Загальний розмір: {full_df.shape}")
    return full_df

if __name__ == "__main__":
    df = load_all_data(base_path='data/raw')
    print("\n--- Перевірка типів даних ---")
    print(df.info())
    print("\n--- Перевірка перших 5 рядків ---")
    print(df.head())
