import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from collections import deque
import json
import os
import random

# ==========================================
# 1. КОПІЯ КЛАСУ GNSSGuard (Щоб все працювало автономно)
# ==========================================
class GNSSGuard:
    def __init__(self, model_path='gnss_model.json', config_path='config.json'):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.features = self.config['input_features']
        self.window_sec = self.config['smoothing_window_seconds']
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.history = {
            'cnoMean': deque(maxlen=200),
            'sat_efficiency': deque(maxlen=200),
            'predictions': deque(maxlen=self.window_sec * 10)
        }

    def _calculate_rolling(self, metric_name, window_seconds):
        data = list(self.history[metric_name])
        if len(data) < 2: return 0.0, 0.0
        relevant_data = data[-window_seconds:] 
        return np.mean(relevant_data), np.std(relevant_data)

    def process_measurement(self, raw_data):
        # Feature Engineering
        num_sv = raw_data.get('numSV', 0)
        num_tracked = raw_data.get('numSatsTracked', 1) 
        sat_eff = np.clip(num_sv / max(1, num_tracked), 0, 5)
        cno_mean = raw_data.get('cnoMean', 0)
        
        self.history['cnoMean'].append(cno_mean)
        self.history['sat_efficiency'].append(sat_eff)
        
        # Features calculation
        cno_mean_5s, cno_std_5s = self._calculate_rolling('cnoMean', 5)
        cno_mean_10s, cno_std_10s = self._calculate_rolling('cnoMean', 10)
        sat_eff_5s, sat_eff_std_5s = self._calculate_rolling('sat_efficiency', 5)
        sat_eff_10s, sat_eff_std_10s = self._calculate_rolling('sat_efficiency', 10)
        
        cno_lag1 = self.history['cnoMean'][-2] if len(self.history['cnoMean']) > 1 else cno_mean
        sat_eff_lag1 = self.history['sat_efficiency'][-2] if len(self.history['sat_efficiency']) > 1 else sat_eff

        input_vector = {
            "cnoMean": cno_mean, "cnoStd": raw_data.get('cnoStd', 0),
            "numSV": num_sv, "numSatsTracked": num_tracked, "sat_efficiency": sat_eff,
            "cnoMean_rolling_mean_5s": cno_mean_5s, "cnoMean_rolling_std_5s": cno_std_5s,
            "cnoMean_rolling_mean_10s": cno_mean_10s, "cnoMean_rolling_std_10s": cno_std_10s,
            "sat_efficiency_rolling_mean_5s": sat_eff_5s, "sat_efficiency_rolling_std_5s": sat_eff_std_5s,
            "sat_efficiency_rolling_mean_10s": sat_eff_10s, "sat_efficiency_rolling_std_10s": sat_eff_std_10s,
            "cnoMean_lag1": cno_lag1, "sat_efficiency_lag1": sat_eff_lag1
        }
        
        features_ordered = [input_vector[f] for f in self.features]
        input_array = np.array([features_ordered], dtype=np.float32)
        raw_pred = self.model.predict(input_array)[0]
        
        self.history['predictions'].append(raw_pred)
        window_size = min(len(self.history['predictions']), self.window_sec)
        smoothed_score = np.mean(list(self.history['predictions'])[-window_size:])
        
        return {"risk_score": float(smoothed_score), "raw_score": float(raw_pred)}

# ==========================================
# 2. ЗАПУСК СИМУЛЯЦІЇ І ЗБІР ДАНИХ
# ==========================================
if __name__ == "__main__":
    # Налаштування шляхів
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'gnss_model.json')
    config_path = os.path.join(script_dir, 'config.json')

    if not os.path.exists(config_path):
        print(f"❌ Config not found. Please put this script next to config.json")
        exit()

    guard = GNSSGuard(model_path=model_path, config_path=config_path)
    
    # Збираємо історію для графіка
    history = []
    
    print("⏳ Running Simulation for Graph...")
    
    for i in range(40): # 40 секунд симуляції
        # Сценарій JAMMING
        if i < 10:
            # Норма (перші 10с)
            data = {'cnoMean': 43.0 + random.uniform(-1, 1), 'cnoStd': 2.0, 'numSV': 25, 'numSatsTracked': 30}
            state = "Normal"
        else:
            # Атака (з 10-ї секунди)
            data = {'cnoMean': 10.0 + random.uniform(-2, 2), 'cnoStd': 8.0, 'numSV': 2, 'numSatsTracked': 30}
            state = "Attack"
            
        result = guard.process_measurement(data)
        
        # Запис
        record = {
            'time': i,
            'cno': data['cnoMean'],
            'raw_score': result['raw_score'],
            'risk_score': result['risk_score'],
            'threshold': 0.1,
            'state': state
        }
        history.append(record)

    df_res = pd.DataFrame(history)

    # ==========================================
    # 3. ВІЗУАЛІЗАЦІЯ
    # ==========================================
    plt.style.use('bmh') # Стиль для наукових графіків
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Графік 1: Вхідний сигнал (Фізика) ---
    ax1.plot(df_res['time'], df_res['cno'], color='#2c3e50', label='Signal Strength (CNO)', linewidth=2)
    ax1.axvline(x=10, color='red', linestyle='--', alpha=0.5, label='Attack Start (Jamming)')
    ax1.fill_between(df_res['time'], 0, df_res['cno'], alpha=0.1, color='#2c3e50')
    ax1.set_ylabel('CNO (dBHz)')
    ax1.set_title('Input: GNSS Signal Degradation', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # --- Графік 2: Реакція моделі (Мозок) ---
    # Сирий прогноз (шумний)
    ax2.plot(df_res['time'], df_res['raw_score'], color='gray', alpha=0.4, label='Raw Model Output (Instant)', linestyle=':')
    # Згладжений ризик (стабільний)
    ax2.plot(df_res['time'], df_res['risk_score'], color='#e74c3c', label='Smoothed Risk Score (Control Signal)', linewidth=3)
    # Поріг
    ax2.axhline(y=0.1, color='green', linestyle='--', linewidth=2, label='Alert Threshold (0.1)')
    ax2.axvline(x=10, color='red', linestyle='--', alpha=0.5)
    
    # Знаходимо точку перетину (час детекції)
    detection_time = df_res[df_res['risk_score'] > 0.1]['time'].min()
    if not pd.isna(detection_time):
        ax2.scatter([detection_time], [0.1], color='red', s=100, zorder=5)
        ax2.annotate(f'Detection\nT={detection_time}s', 
                     xy=(detection_time, 0.1), xytext=(detection_time+2, 0.2),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    ax2.set_ylabel('Degradation Score (0-1)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_title('Output: Attack Detection & Filtering', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 0.3) # Фокусуємось на нижній частині графіка, де поріг
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Збереження
    save_path = os.path.join(script_dir, 'simulation_result.png')
    plt.savefig(save_path, dpi=300)
    print(f"✅ Graph saved to: {save_path}")
    plt.show()
