import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set plot style
sns.set_theme(style="whitegrid")

# ==========================================
# STAGE 1: DATASET SIMULATION
# ==========================================
print("--- Stage 1: Creating Dataset ---")
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=8760, freq='h')
df = pd.DataFrame({'Date': dates})

df['Hour'] = df['Date'].dt.hour
df['Month'] = df['Date'].dt.month

# Weather simulation
df['Ambient_Temp'] = 15 + 12 * np.sin((df['Hour'] - 6) * np.pi / 12) + np.random.normal(0, 1.5, 8760)
df['Cloud_Cover_Pct'] = np.random.randint(0, 100, 8760)

# Solar Irradiance Model
base_irradiance = 900 * np.maximum(0, np.sin((df['Hour'] - 5) * np.pi / 13))
df['Irradiance_W_m2'] = base_irradiance * (1 - 0.75 * df['Cloud_Cover_Pct'] / 100)

# PV Panel Temperature (Irradiance heats up the panel more than the ambient air!)
df['Module_Temp'] = df['Ambient_Temp'] + (df['Irradiance_W_m2'] * 0.03)

# TARGET VARIABLE: Power Generation (kW)
# Efficiency drops by 0.4% per degree when panel temperature exceeds 25°C (Engineering reality)
temp_loss_factor = np.maximum(0, df['Module_Temp'] - 25) * 0.004
df['Power_Generation_kW'] = df['Irradiance_W_m2'] * 0.18 * (1 - temp_loss_factor) + np.random.normal(0, 2, 8760)

# Strictly set nighttime generation to zero
df.loc[df['Irradiance_W_m2'] < 5, 'Power_Generation_kW'] = 0
print(f"Dataset successfully created. Shape: {df.shape}\n")


# ==========================================
# STAGE 2: PREVIEW AND EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("--- Stage 2: Data Preview and EDA ---")
print(df[['Ambient_Temp', 'Irradiance_W_m2', 'Module_Temp', 'Power_Generation_kW']].describe())

# Correlation Matrix (How much each feature affects the target?)
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Ambient_Temp', 'Irradiance_W_m2', 'Module_Temp', 'Power_Generation_kW']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Relationships (Correlation Matrix)")
plt.savefig('correlation_matrix.png')
plt.close()
print("-> Correlation matrix saved as 'correlation_matrix.png'.\n")


# ==========================================
# STAGE 3: FEATURE ENGINEERING
# ==========================================
print("--- Stage 3: Feature Engineering ---")
# Teach the model the cyclic nature of time using Sin/Cos transformations
df['Hour_Sin'] = np.sin(df['Hour'] * (2 * np.pi / 24))
df['Hour_Cos'] = np.cos(df['Hour'] * (2 * np.pi / 24))

# Lag Feature: Power generation from 1 hour ago (Very powerful for time-series forecasting)
df['Power_Lag_1h'] = df['Power_Generation_kW'].shift(1)
# Backfill the first row since it will be NaN
df['Power_Lag_1h'] = df['Power_Lag_1h'].bfill()

print("Newly added features: Hour_Sin, Hour_Cos, Power_Lag_1h\n")


# ==========================================
# STAGE 4: MODEL TRAINING AND COMPARISON
# ==========================================
print("--- Stage 4: Model Training ---")
# Define Inputs and Target
features = ['Ambient_Temp', 'Cloud_Cover_Pct', 'Irradiance_W_m2', 'Module_Temp', 'Hour_Sin', 'Hour_Cos', 'Power_Lag_1h']
X = df[features]
y = df['Power_Generation_kW']

# Train/Test Split (80% Train, 20% Test)
# Note: shuffle=False is highly recommended for time-series data to avoid data leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# 2. Polynomial Regression (Degree = 2)
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
poly_pred = poly_reg.predict(X_test_poly)

# 3. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)


# ==========================================
# STAGE 5: EVALUATION AND FUTURE FORECASTING
# ==========================================
print("--- Stage 5: Model Evaluation ---")

def model_metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{name}] MAE: {mae:.2f} kW | RMSE: {rmse:.2f} kW | R² Score: {r2:.4f}")

model_metrics(y_test, lr_pred, "Linear Regression")
model_metrics(y_test, poly_pred, "Polynomial Regression")
model_metrics(y_test, rf_pred, "Random Forest")

# Future Forecasting: Visualize a 24-hour window from the test set
plt.figure(figsize=(12, 6))
plt.plot(y_test.values[100:124], label='Actual Generation', color='black', linewidth=2)
plt.plot(rf_pred[100:124], label='Random Forest Prediction', color='orange', linestyle='--')
plt.plot(lr_pred[100:124], label='Linear Regression Prediction', color='blue', linestyle=':')
plt.title("24-Hour Solar Power Generation Forecast vs Actual Values")
plt.xlabel("Hour")
plt.ylabel("Generation (kW)")
plt.legend()
plt.savefig('forecast_comparison.png')
plt.close()
print("\n-> Forecast plot successfully saved as 'forecast_comparison.png'.")
