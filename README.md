# Структура проєкту

```text
gnss_challenge/
│
├── data/                      <-- Тут живуть дані
│   ├── raw/                   <-- Оригінальні CSV файли (НЕ ЗМІНЮВАТИ!)
│   │   ├── 2025-08-01 11-23-55.csv
│   │   ├── ... (інші файли)
│   └── processed/             <-- Очищені дані
│
├── notebooks/                 <-- Jupyter Notebooks для експериментів
│   ├── 01_eda_analiz_danih.ipynb   <-- Графіки, статистика
│   ├── 02_baseline_model.ipynb     <-- Перша проста модель
│   └── 03_xgboost_tuning.ipynb     <-- Основна складна модель
│
├── src/                       <-- Чистий код (скрипти)
│   ├── __init__.py            <-- (порожній файл, щоб Python бачив це як пакет)
│   ├── data_loader.py         <-- Функція завантаження даних
│   ├── features.py            <-- Генерація фіч
│   └── train.py               <-- Скрипт для тренування фінальної моделі
│
├── models/                    <-- Збережені навчені моделі
│   ├── baseline_logreg.pkl
│   └── xgboost_v1.json
│
├── reports/                   <-- Результати для презентації
│   ├── figures/               <-- Графіки (png, jpg)
│   └── final_report.md        <-- Чернетка документації
│
├── .gitignore                 <-- Список того, що НЕ лити на GitHub
├── requirements.txt           <-- Список бібліотек
└── README.md                  <-- Опис проєкту
