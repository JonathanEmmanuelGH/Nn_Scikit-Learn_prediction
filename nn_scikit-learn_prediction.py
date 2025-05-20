# nn_scikit-learn_prediction.py
# ===============================================================================
# Title: Neural NEtwork for Order PRediction and Descriptive Analysis
# Author: Jonathan Emmanuel García Hernández
# Created for: Apprenticeship project addapted to Github Portfolio Showcase
# Descrpition: Predicts target variable using a 3 capes neural network
#                and visualizes pattern over time.
# ================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU

# Load and Prepare Data
df = pd.read_parquet('jonathan_dataframe.parquet')

# + Date features
df['Date'] = pd.to_datetime(df['Date'])
df['Date_month'] = df['Date'].dt.month
df['Date_dow'] = df['Date'].dt.dayofweek
df['Date_weeknd'] = (df['Date_dow'] >= 5).astype(int)
df['sin_month'] = np.sin(2 * np.pi * df['Date_month'] / 12)
df['cos_month'] = np.cos(2 * np.pi * df['Date_month'] / 12)
df['sin_dow'] = np.sin(2 * np.pi * df['Date_dow'] / 7)
df['cos_dow'] = np.sin(2 * np.pi * df['Date_dow'] / 7)

# Feature Selection
cate_var = ['id_a', 'ag', 'a1', 'a2', 'a3', 'a4', 'a5']
bin_var = ['Date_weeknd']
cycl_var = ['sin_month', 'cos_month', 'sin_dow', 'cos_dow']

cate_transf = OneHotEncoder()
preprocessor = ColumnTransformer([
    ('cat', cate_transf, cate_var),
    ('bin', 'passthrough', bin_var),
    ('cyc', 'passthrough', cycl_var)
])

X = preprocessor.fit_transform(df)
y = df['a6'].values

# Train/Test Split & Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=12)

y_scaler = StandardScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

# Neural Network Model
Model_1LR = Sequential()
Model_1LR.add(Dense(32, input_shape=(X_train.shape[1],)))
Model_1LR.add(LeakyReLU(alpha=0.01))
Model_1LR.add(Dense(16))
Model_1LR.add(LeakyReLU(alpha=0.01))
Model_1LR.add(Dense(1, activation='linear'))

Model_1LR.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train Model
Model_1LR.fit(X_train, y_train_scaled, epochs=15, batch_size=40)

# Evaluate Model
loss, MAE = Model_1LR.evaluate(X_test, y_test_scaled)
print(f'\nModel Evaluation:')
print(f'Loss (MSE scaled): {loss:.4f}')
print(f'MAE (scaled): {MAE:.4f}')

y_pred_scaled = Model_1LR.predict(X_test).flatten()
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

print(f'\nPerformance on Original Scale:')
print(f'R² Score: {r2_score(y_test, y_pred):.3f}')
print(f'MAE: {mean_absolute_error(y_test, y_pred):.3f}')
print(f'MSE: {mean_squared_error(y_test, y_pred):.3f}')

# Visualization

# + Orders per day
df_sorted = df.sort_values(by='Date')
df_count_total = df_sorted['Date'].value_counts().sort_index()

plt.figure(figsize=(20, 6))
plt.plot(df_count_total.index, df_count_total.values, linewidth=1)
plt.title('📦 Total Orders per Day')
plt.xlabel('Date')
plt.ylabel('Order Count')
plt.grid(True)
plt.tight_layout()
plt.show()

# + Average orders per month
n_years = df['Date'].dt.year.nunique()
avg_order_month = df.groupby('Date_month').size() / n_years

plt.figure(figsize=(20, 6))
plt.plot(avg_order_month.index, avg_order_month.values, linewidth=1)
plt.title('📊 Average Monthly Orders (Across Years)')
plt.xlabel('Month')
plt.ylabel('Avg Order Count')
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.show()

# + True vs Predicted (filtered)
residuals = y_test - y_pred
mask = (y_test < 4000) & (y_pred < 4000)
y_test_notoutliers = y_test[mask]
y_pred_notoutliers = y_pred[mask]
residuals_notoutliers = residuals[mask]

plt.figure(figsize=(8, 6))
plt.scatter(y_test_notoutliers, y_pred_notoutliers, alpha=0.3)
plt.plot([y_test_notoutliers.min(), y_test_notoutliers.max()],
         [y_test_notoutliers.min(), y_test_notoutliers.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('🔍 True vs Predicted (No Outliers)')
plt.grid(True)
plt.tight_layout()
plt.show()

# + Residual Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_notoutliers, residuals_notoutliers, alpha=0.3)
plt.hlines(0, xmin=y_test_notoutliers.min(), xmax=y_test_notoutliers.max(), colors='r', linestyles='--')
plt.xlabel('True Values')
plt.ylabel('Residuals')
plt.title('📉 Residuals (No Outliers)')
plt.grid(True)
plt.tight_layout()
plt.show()
