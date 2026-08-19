# ☀️ Solar Energy Prediction using Forecasting Dataset

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)

> **Short Description:** A machine learning forecasting project that predicts solar power generation (`Power_Generation_kW`) based on 1-year weather dynamics (irradiance, cloudiness, temperatures) and time-series feature engineering to optimize energy efficiency and grid operation.

---

## 1. Introduction

* **Definition:** This study leverages 1-year hourly meteorological parameters—including ambient temperature, solar irradiance, module temperature, and lagged generation metrics—to build accurate prediction models for solar panel output.

---

## 2. Overview

* **What is the Problem?**
  Unpredictable renewable energy output challenges grid stability and operational planning. Predicting solar power generation in advance allows energy providers to optimize consumption efficiency, prepare backup resources, and maximize profit margins. Additionally, identifying the most efficient ML algorithm for this specific data structure is a core goal.

* **How to Solve It?**
  Using the `scikit-learn` framework, multiple regression algorithms (**Linear Regression**, **Polynomial Regression**, and **Random Forest Regressor**) were implemented. Their performances were benchmarked against test data to select the optimal model.

* **Why Choose This Method?**
  Strategic energy forecasting requires scalable and accessible machine learning pipelines. By employing robust standard libraries, this project delivers a clear, reproducible benchmark for real-world solar energy management.

---

## 3. System Architecture & Key Features

### 3.1 System Architecture

<div align="center">
  <img width="750" alt="system_arch_p7" src="https://github.com/user-attachments/assets/3110a74c-5c00-4288-921f-ac2e869668ce" />
</div>

### 3.2 Key Features

* **Raw Data & Preprocessing:** 1-year dataset containing 8,760 entries (`Ambient_Temp`, `Irradiance_W_m2`, `Module_Temp`, `Power_Generation_kW`).
* **Feature Engineering:**
  * **Cyclical Time Encoding:** `Hour_Sin` and `Hour_Cos` transformations for 24-hour daily cycles.
  * **Autoregressive Feature:** `Power_Lag_1h` (1-hour delayed power generation values) to capture time-series autocorrelation.
* **Model Training:** Comparative evaluation across linear and ensemble models.

---

## 4. Results & Performance Metrics

### 4.1 Terminal Output

<div align="center">
  <img width="700" alt="Terminal Execution Results" src="https://github.com/user-attachments/assets/11ce4e87-f199-4324-85ec-1df0cb1c0c5c" />
</div>

<br>

### 4.2 Benchmark Table

| Model Algorithm | MAE (kW) | RMSE (kW) | R² Score | Performance Verdict |
| :--- | :---: | :---: | :---: | :--- |
| **Linear Regression** | 1.33 kW | 1.87 kW | 0.9978 | Baseline performance |
| **Polynomial Regression** | **0.82 kW** | **1.41 kW** | **0.9988** | **Best Performing Model** |
| **Random Forest Regressor** | 0.88 kW | 1.56 kW | 0.9985 | Strong ensemble baseline |

> **Key Takeaway:** Incorporating the 1-hour lag feature (`Power_Lag_1h`) significantly boosted overall accuracy ($R^2 > 0.99$). Polynomial Regression achieved the lowest error metrics (MAE: 0.82 kW), proving to be the most optimal model for capturing the subtle non-linear thermodynamics of the panels.

---

## 5. Artifacts & Outputs

The execution pipeline automatically generates and stores evaluation visuals:

* `correlation_matrix.png`: Exploratory data analysis showing feature correlations.
* `forecast_comparison.png`: Time-series overlay plot comparing real vs. predicted power outputs across all models.

---

## 6. Getting Started

### 6.1 Prerequisites
* Python 3.8+
* `pip` package manager

### 6.2 Installation & Execution

```bash
# Clone the repository
git clone [https://github.com/your-username/solar-energy-prediction.git](https://github.com/your-username/solar-energy-prediction.git)
cd solar-energy-prediction

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Run the main pipeline
python3 main.py
