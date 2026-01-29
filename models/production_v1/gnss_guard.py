import json
import numpy as np
import xgboost as xgb
from collections import deque
import time
import os
import random  # <--- Ось чого не вистачало!

class GNSSGuard:
    def __init__(self, model_path='gnss_model.json', config_path='config.json'):
        # 1. Завантаження конфігурації
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.features = self.config['input_features']
        self.window_sec = self.config['smoothing_window_seconds']
        
        # 2. Завантаження моделі
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        
        # 3. Ініціалізація буферів
        self.history = {
            'cnoMean': deque(maxlen=200),
            'sat_efficiency': deque(maxlen=200),
            'predictions': deque(maxlen=self.window_sec * 10)
        }
        
        print(f"✅ GNSS Guard v{self.config['version']} loaded.")
        print(f"   Window: {self.window_sec}s | Thresholds: Safe < {self.config['threshold_safe']} < Fail")

    def _calculate_rolling(self, metric_name, window_seconds):
        data = list(self.history[metric_name])
        if len(data) < 2:
            return 0.0, 0.0
        relevant_data = data[-window_seconds:] 
        return np.mean(relevant_data), np.std(relevant_data)

    def process_measurement(self, raw_data):
        # 1. Feature Engineering
        num_sv = raw_data.get('numSV', 0)
        num_tracked = raw_data.get('numSatsTracked', 1) 
        sat_eff = np.clip(num_sv / max(1, num_tracked), 0, 5)
        cno_mean = raw_data.get('cnoMean', 0)
        cno_std = raw_data.get('cnoStd', 0)
        
        self.history['cnoMean'].append(cno_mean)
        self.history['sat_efficiency'].append(sat_eff)
        
        cno_mean_5s, cno_std_5s = self._calculate_rolling('cnoMean', 5)
        cno_mean_10s, cno_std_10s = self._calculate_rolling('cnoMean', 10)
        sat_eff_5s, sat_eff_std_5s = self._calculate_rolling('sat_efficiency', 5)
        sat_eff_10s, sat_eff_std_10s = self._calculate_rolling('sat_efficiency', 10)
        
        cno_lag1 = self.history['cnoMean'][-2] if len(self.history['cnoMean']) > 1 else cno_mean
        sat_eff_lag1 = self.history['sat_efficiency'][-2] if len(self.history['sat_efficiency']) > 1 else sat_eff

        # 2. Vector Assembly
        input_vector = {
            "cnoMean": cno_mean,
            "cnoStd": cno_std,
            "numSV": num_sv,
            "numSatsTracked": num_tracked,
            "sat_efficiency": sat_eff,
            "cnoMean_rolling_mean_5s": cno_mean_5s,
            "cnoMean_rolling_std_5s": cno_std_5s,
            "cnoMean_rolling_mean_10s": cno_mean_10s,
            "cnoMean_rolling_std_10s": cno_std_10s,
            "sat_efficiency_rolling_mean_5s": sat_eff_5s,
            "sat_efficiency_rolling_std_5s": sat_eff_std_5s,
            "sat_efficiency_rolling_mean_10s": sat_eff_10s,
            "sat_efficiency_rolling_std_10s": sat_eff_std_10s,
            "cnoMean_lag1": cno_lag1,
            "sat_efficiency_lag1": sat_eff_lag1
        }
        
        try:
            features_ordered = [input_vector[f] for f in self.features]
        except KeyError as e:
            return {"is_attack": False, "message": f"MISSING FEATURE: {e}"}

        # 3. Prediction
        input_array = np.array([features_ordered], dtype=np.float32)
        raw_pred = self.model.predict(input_array)[0]
        
        # 4. Smoothing
        self.history['predictions'].append(raw_pred)
        window_size = min(len(self.history['predictions']), self.window_sec)
        smoothed_score = np.mean(list(self.history['predictions'])[-window_size:])
        
        # 5. Result
        is_attack = smoothed_score > 0.5
        return {
            "is_attack": is_attack,
            "risk_score": float(smoothed_score),
            "raw_score": float(raw_pred),
            "message": "⚠️ JAMMING DETECTED" if is_attack else "✅ GNSS OK"
        }

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'gnss_model.json')
    config_path = os.path.join(script_dir, 'config.json')

    if not os.path.exists(config_path):
        print(f"❌ Config not found in {script_dir}")
        exit()

    guard = GNSSGuard(model_path=model_path, config_path=config_path)
    
    # -------------------------------------------------------------
    # 🔧 КАЛІБРУВАННЯ: Встановлюємо чутливий поріг
    # 0.1 = Початок деградації (точність гірша за 10 метрів)
    # -------------------------------------------------------------
    ALERT_THRESHOLD = 0.1 
    
    print("\n--- Starting CALIBRATED Simulation (Threshold = 0.1) ---")
    
    for i in range(25):
        # 1. Сценарій
        if i < 5:
            # ✅ НОРМА
            current_data = {
                'cnoMean': 43.0 + random.uniform(-2, 2),
                'cnoStd': 2.0 + random.uniform(0, 1),
                'numSV': 25,
                'numSatsTracked': 30
            }
            scenario = "✅ NORMAL "
        else:
            # 🔴 JAMMING
            current_data = {
                'cnoMean': 10.0 + random.uniform(-2, 2),
                'cnoStd': 8.0 + random.uniform(0, 4),
                'numSV': 2,
                'numSatsTracked': 30
            }
            scenario = "🔴 JAMMING"

        # 2. Обробка
        result = guard.process_measurement(current_data)
        
        # 3. Прийняття рішення (З новим порогом!)
        risk = result['risk_score']
        is_attack = risk > ALERT_THRESHOLD  # <--- ВИКОРИСТОВУЄМО НОВИЙ ПОРІГ
        
        # 4. Вивід
        status_icon = "🔴" if is_attack else "🟢"
        color = "\033[91m" if is_attack else "\033[92m"
        reset = "\033[0m"
        msg = "⚠️ JAMMING DETECTED" if is_attack else "✅ GNSS OK"
        
        print(f"T={i:02}s | {scenario} | Raw:{result['raw_score']:.3f} -> Score:{risk:.3f} | {status_icon} {color}{msg}{reset}")
        
        time.sleep(1.0)