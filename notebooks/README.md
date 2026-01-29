# 📚 Notebooks Structure

## 🎯 Основні Jupyter Notebooks (для диплома)

| # | Файл | Опис |
|---|------|------|
| 01 | `01_EDA_and_Physics.ipynb` | Розвідувальний аналіз даних + фізика GNSS сигналу |
| 02a | `02_Robust_3D_GNSS_Model.ipynb` | **ГОЛОВНА МОДЕЛЬ**: XGBoost з 3D Spatial Error + Anti-Flickering |
| 02b | `02_Adaptive_3D_GNSS_Model.ipynb` | Адаптивна модель з авто-кореляційним аналізом |
| 03 | `03_Leakage_Ablation_Study.ipynb` | Перевірка на Data Leakage (Physics-only vs All features) |
| 05 | `05_Attack_Clustering_Analysis.ipynb` | K-Means кластеризація: Jamming vs Spoofing |
| 06 | `06_Transition_Dynamics_Analysis.ipynb` | Аналіз гістерезису: швидкість деградації vs відновлення |
| 24 | `24_Diploma_Charts_Generator.ipynb` | Генератор всіх графіків для диплому (300 DPI) |

## 📁 Структура папок

```
notebooks/
├── 01_EDA_and_Physics.ipynb          # EDA + Фізика
├── 02_Robust_3D_GNSS_Model.ipynb     # ГОЛОВНА МОДЕЛЬ
├── 02_Adaptive_3D_GNSS_Model.ipynb   # Адаптивна модель
├── 03_Leakage_Ablation_Study.ipynb   # Ablation study
├── 05_Attack_Clustering_Analysis.ipynb
├── 06_Transition_Dynamics_Analysis.ipynb
├── 24_Diploma_Charts_Generator.ipynb
├── figures/                          # Вихідні графіки
├── scripts/                          # Допоміжні скрипти (пусто)
└── archive/                          # Архів старих .py файлів
```

## 🔑 Ключові концепції моделі

### 3D Spatial Error
```python
spatial_error_3d = sqrt(hAcc² + vAcc²)
```

### Soft Target (Anti-Flickering)
```python
# Розширення зон атаки для стабільності
target_soft = label.rolling(5, center=True).max().fillna(label)
```

### Hysteresis Post-Processing
```python
# Гістерезис: включення > 0.8, виключення < 0.4
LOW_THRESHOLD = 0.4
HIGH_THRESHOLD = 0.8
```

## 📊 Порядок виконання

1. **Спочатку**: `01_EDA_and_Physics.ipynb` — зрозуміти дані
2. **Основний аналіз**: `02_Robust_3D_GNSS_Model.ipynb` — тренування моделі
3. **Валідація**: `03_Leakage_Ablation_Study.ipynb` — перевірка чесності
4. **Додатковий аналіз**:
   - `05_Attack_Clustering_Analysis.ipynb` — типи атак
   - `06_Transition_Dynamics_Analysis.ipynb` — динаміка переходів
5. **Фіналізація**: `24_Diploma_Charts_Generator.ipynb` — графіки для звіту

## 🗃️ Архів

Папка `archive/` містить старі .py скрипти, які були конвертовані в Jupyter notebooks або стали непотрібними. Зберігаються для reference.
