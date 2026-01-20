# Benchmarking Air Quality Data using Scientific Machine Learning Methods

A reproducible **Scientific Machine Learning (SciML)** benchmarking suite for **multi-horizon AQI forecasting**
using **physics-guided deep learning and baseline statistical models**. This module proposed **MLP+Physics** and **LSTM+Physics**
for predicting AQI at multiple horizons (**LAG = 1, 7, 14, 30**) using daily **PM₂.₅** and **Ozone**
USA Environmental Protection Agency (EPA) datasets (2022–2024) and set a **benchmark** by comparing other statistical and deep learning models.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [GitHub Setup Instructions](#github-setup-instructions)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)
  - [3. Install Dependencies](#3-install-dependencies)
  - [4. Run the Notebooks](#4-run-the-notebooks)
- [Project Folder Structure](#project-folder-structure)
- [Datasets](#datasets)
- [How It Works](#how-it-works)
- [Benchmarked Notebooks](#benchmarked-notebooks)
- [Metrics](#metrics)
- [Outputs](#outputs)
- [Final Notes](#final-notes)
- [Copyright](#copyright)

---

## Overview

This benchmarking module evaluates **physics-guided learning** for AQI forecasting:

- Forecasting target: `AQI(t + LAG)`
- Forecast horizons: `LAG ∈ {1, 7, 14, 30}`
- Pollutants:
  - **PM₂.₅ AQI**
  - **Ozone AQI**

### Physics-Guided Objective

A physics-derived AQI is computed from pollutant concentration using a **piecewise breakpoint mapping**
(used as a physical constraint during training).

Training uses a composite loss:

- Total Loss = (lambda_data × Data Loss) + (lambda_phys × Physics Loss)

Where:
- Data Loss = MSE(predicted AQI, true AQI)
- Physics Loss = MSE(predicted AQI, physics-derived AQI)

---

## Features

- Multi-horizon AQI forecasting (**LAG 1 / 7 / 14 / 30**)
- Physics-guided deep learning models:
  - **MLP + Physics**
  - **LSTM + Physics**
- Chronological split (no shuffle) to avoid temporal leakage
- Figures saved under:
  - `analysis/images/`
  - `models/images/`

---

## Tech Stack

- Python 3.10+
- PyTorch
- NumPy, Pandas
- scikit-learn
- Matplotlib
- Statsmodels (SARIMAX)
- Jupyter Notebook

---

## GitHub Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/d3lab-unt/AQ_SciML.git
cd AQ_SciML/sciML_benchmarking
```

### 2. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks
```bash
jupyter notebook
```
---

## Project Folder Structure

```bash
sciML_benchmarking/
├── analysis/
│   ├── images/                               # Dataset analysis figures
│   ├── Dataset_Preparation.ipynb             # Lag-wise dataset construction
│   ├── Dataset_analysis.ipynb                # Exploratory analysis + visualizations
│   └── Readme.md
├── data/
│   ├── PM2.5CombinedAqi_2022-2024.csv         # Base PM2.5 dataset
│   ├── Ozone_combined_aqi_2022-2024.csv       # Base Ozone dataset
│   ├── LAG1_PM_Combined_AQI_2022_2024.csv     # PM supervised dataset (LAG 1)
│   ├── LAG7_PM_Combined_AQI_2022_2024.csv     # PM supervised dataset (LAG 7)
│   ├── LAG14_PM_Combined_AQI_2022_2024.csv    # PM supervised dataset (LAG 14)
│   ├── LAG30_PM_Combined_AQI_2022_2024.csv    # PM supervised dataset (LAG 30)
│   ├── LAG1_Ozone_Combined_AQI_2022_2024.csv  # Ozone supervised dataset (LAG 1)
│   ├── LAG7_Ozone_Combined_AQI_2022_2024.csv  # Ozone supervised dataset (LAG 7)
│   ├── LAG14_Ozone_Combined_AQI_2022_2024.csv # Ozone supervised dataset (LAG 14)
│   ├── LAG30_Ozone_Combined_AQI_2022_2024.csv # Ozone supervised dataset (LAG 30)
│   └── Readme.md
├── models/
│   ├── images/                               # Model result plots
│   ├── MLP_PHYSICS_PM.ipynb                  # MLP + Physics experiments (PM2.5)
│   ├── MLP_PHYSICS_OZONE.ipynb               # MLP + Physics experiments (Ozone)
│   ├── LSTM_PHYSICS_PM.ipynb                 # LSTM + Physics experiments (PM2.5)
│   ├── LSTM_PHYSICS_OZONE.ipynb              # LSTM + Physics experiments (Ozone)
│   ├── Linear_Model_Prediction_PM2_5.ipynb    # Linear regression baseline (PM2.5)
│   ├── Linear_Model_Prediction_Ozone.ipynb    # Linear regression baseline (Ozone)
│   ├── SARIMAX_Prediction_PM2_5.ipynb         # SARIMAX baseline (PM2.5)
│   ├── SARIMAX_Prediction_Ozone.ipynb         # SARIMAX baseline (Ozone)
│   └── Readme.md
└── Readme.md                                 # Main project README
```
---
## Datasets

All datasets are stored under the `data/` directory.  
The raw air quality data were collected from the **United States Environmental Protection Agency (EPA)** official source:

- EPA Website: https://www.epa.gov/

This project focuses on two pollutants:
- **PM₂.₅ (Particulate Matter 2.5)**
- **Ozone (O₃)**

The overall dataset preparation pipeline follows a reproducible two-stage procedure:
1) **Multi-year station-averaged dataset construction (2022–2024)**  
2) **Lag-wise (multi-horizon) supervised dataset generation**

### Base Datasets (Combined Multi-Year)

The combined base datasets contain daily values (2022–2024):

- `PM2.5CombinedAqi_2022-2024.csv`
- `Ozone_combined_aqi_2022-2024.csv`

These combined datasets were created by:
- loading EPA daily CSV data for each year (2022, 2023, 2024)
- parsing the date field
- removing missing rows
- sorting by date
- averaging daily values across all monitoring stations

#### Key Columns (Base Dataset)
Each base dataset contains at minimum:
- `DATE` — Daily timestamp
- `Daily_Mean_PM` or `Daily_Mean_Ozone` — Daily station-averaged pollutant concentration
- `Daily_AQI_Value` — Daily station-averaged AQI value

### Lag-wise Supervised Datasets (Multi-Horizon Forecasting)

Lag-wise datasets were constructed for multi-horizon AQI forecasting:

#### PM₂.₅ Lag Datasets
- `LAG1_PM_Combined_AQI_2022_2024.csv`
- `LAG7_PM_Combined_AQI_2022_2024.csv`
- `LAG14_PM_Combined_AQI_2022_2024.csv`
- `LAG30_PM_Combined_AQI_2022_2024.csv`

#### Ozone Lag Datasets
- `LAG1_Ozone_Combined_AQI_2022_2024.csv`
- `LAG7_Ozone_Combined_AQI_2022_2024.csv`
- `LAG14_Ozone_Combined_AQI_2022_2024.csv`
- `LAG30_Ozone_Combined_AQI_2022_2024.csv`

Each lag-wise dataset includes the forecasting target:

- `AQI_Targeted_Value_LAG_k`

where `k ∈ {1, 7, 14, 30}` represents the prediction horizon in days.

### Dataset Preparation

This dataset preparation follows the steps below.

#### Stage 1 — Multi-Year Station-Averaged EPA Dataset

For each year `y ∈ {2022, 2023, 2024}`:

1. Load the EPA daily CSV file for that year  
2. Convert the `Date` column into datetime  
3. Drop invalid rows with missing values in:  
   - Date  
   - Pollutant concentration (PM₂.₅ or Ozone)  
   - AQI  
4. Sort all records chronologically  
5. For every day `t`, compute daily averages across all stations:
   - Daily Mean Concentration: `c̄y(t)`  
   - Daily Mean AQI: `āy(t)`  
6. Store daily aggregated rows as:
   - `(t, c̄y(t), āy(t))`

Finally, concatenate all yearly averaged datasets into one multi-year dataset.

#### Stage 2 — LAG-Based Dataset Construction (Multi-Horizon Supervised Learning)

For a given forecasting horizon `LAG ≥ 1`:

1. Load the combined dataset (from Stage 1)
2. For each day index `i`, define the AQI target using a forward shift:
   - Target AQI at horizon:
     - `AQI_Targeted_Value_LAG = Daily_AQI_Value[i + LAG]`
3. Remove final `LAG` rows with missing targets (`NaN`)
4. Save each horizon dataset separately (LAG1, LAG7, LAG14, LAG30)

---

## How It Works

This benchmarking module evaluates **multi-horizon AQI forecasting** using both **physics-guided deep learning** and **classical baselines**.

### 1) Multi-Horizon Dataset Construction

For each pollutant, supervised learning datasets were created using multiple forecast horizons:

- `AQI_Targeted_Value_LAG_1  = AQI(t + 1)`
- `AQI_Targeted_Value_LAG_7  = AQI(t + 7)`
- `AQI_Targeted_Value_LAG_14 = AQI(t + 14)`
- `AQI_Targeted_Value_LAG_30 = AQI(t + 30)`

Input features are constructed from daily pollutant concentration and current AQI:

- `X(t) = [Pollutant(t), AQI(t)]`
- `y(t) = AQI(t + LAG)`

A chronological split is applied to avoid temporal leakage:
- first 80% samples → training
- last 20% samples → testing
- no shuffling

### 2) Physics-Guided Learning

Physics guidance is applied using a **piecewise pollutant → AQI breakpoint mapping**.

For each horizon, a physics-derived AQI target is computed using the pollutant value at the targeted time step:

- `AQI_phys(t + LAG) = f(Pollutant(t + LAG))`

Training uses a composite loss:

- `Total Loss = (lambda_data × Data Loss) + (lambda_phys × Physics Loss)`

Lambda settings are swept to analyze the effect of physics constraints:

- `(lambda_data, lambda_phys) ∈ {(0.0,1.0), (0.3,0.7), (0.5,0.5), (0.7,0.3), (1.0,0.0)}`

### 3) Model Benchmarking

Models evaluated in this module include:
- `Linear Regression`
- `SARIMAX`
- `MLP`
- `MLP + Physics`
- `LSTM`
- `LSTM + Physics`

Each model is trained and evaluated separately for:
- pollutant type (PM₂.₅ and Ozone)
- forecasting horizon (LAG 1/7/14/30)
- physics weighting configuration (lambda sweep)

### 4) Output Generation

The benchmarking pipeline produces:
- performance summary tables (MAE, RMSE, NMSE)
- visualizations saved under:
  - `analysis/images/`
  - `models/images/`

---

## Benchmarked Notebooks

This project includes notebooks for **dataset preparation**, **exploratory analysis**, **physics-guided deep learning models**, and **baseline forecasting models**.

### Analysis
- `analysis/Dataset_Preparation.ipynb` — Create lag-wise supervised datasets (LAG 1/7/14/30)
- `analysis/Dataset_analysis.ipynb` — Exploratory data analysis and dataset visualization

### Physics-Guided Deep Learning Models
- `models/MLP_PHYSICS_PM.ipynb` — MLP + Physics experiments for **PM₂.₅ AQI**
- `models/LSTM_PHYSICS_PM.ipynb` — LSTM + Physics experiments for **PM₂.₅ AQI**
- `models/MLP_PHYSICS_OZONE.ipynb` — MLP + Physics experiments for **Ozone AQI**
- `models/LSTM_PHYSICS_OZONE.ipynb` — LSTM + Physics experiments for **Ozone AQI**

### Classical Baselines
- `models/Linear_Model_Prediction_PM2_5.ipynb` — Linear Regression baseline for **PM₂.₅**
- `models/Linear_Model_Prediction_Ozone.ipynb` — Linear Regression baseline for **Ozone**
- `models/SARIMAX_Prediction_PM2_5.ipynb` — SARIMAX baseline for **PM₂.₅**
- `models/SARIMAX_Prediction_Ozone.ipynb` — SARIMAX baseline for **Ozone**

---

## Metrics

This benchmarking suite reports the following evaluation metrics for each model and forecasting horizon (LAG 1/7/14/30):

- **MAE (Mean Absolute Error)**
  - Measures average absolute difference between predicted AQI and true AQI.
  - Lower is better.

- **RMSE (Root Mean Squared Error)**
  - Penalizes larger prediction errors more heavily than MAE.
  - Lower is better.

- **NMSE (Normalized Mean Squared Error)**
  - Variance-normalized error to allow fair comparison across horizons.
  - Lower is better.

---

## Outputs

This benchmarking pipeline generates:

### 1) Result Tables
- Model performance summary tables for each pollutant and horizon (LAG 1/7/14/30)
- Metrics included: **MAE**, **RMSE**, **NMSE**
- Output files are stored as CSV (generated from experiments)

### 2) Figures and Visualizations
Plots and analysis figures are saved under:

- `analysis/images/` — dataset preparation and exploratory analysis figures  
- `models/images/` — model prediction plots and performance comparisons

### 3) Notebook Outputs
All experiment results, logs, and visual outputs are reproducible directly by running the notebooks under:

- `analysis/`
- `models/`

---

## Final Notes

This benchmarking suite supports reproducible **Scientific Machine Learning (SciML)** experiments for AQI forecasting by providing:

- Standardized datasets for **PM₂.₅** and **Ozone** (2022–2024)
- Lag-wise multi-horizon supervised learning setup (**LAG 1 / 7 / 14 / 30**)
- Fair and time-aware evaluation using chronological splitting (no shuffle)
- Benchmark comparison between:
  - Physics-guided deep learning (**MLP+Physics**, **LSTM+Physics**)
  - Baselines (**Linear Regression**, **SARIMAX**, **MLP**, **LSTM**)
- Physics-guided learning formulation through a pollutant→AQI breakpoint mapping
- Fully reproducible notebooks with saved results and figures

## Copyright

© 2026 Data Driven Decisions LAB, Data Science Department, University of North Texas. All rights reserved.

This repository and its contents (code, experiments, figures, and documentation) are intended for academic and research purposes.  
No part of this project may be reproduced, redistributed, or used for commercial purposes without explicit permission from the author.

