# -*- coding: utf-8 -*-
"""
Created on Mon May 25 17:42:35 2026

@author: Jams
"""

import pandas as pd
import numpy as np
import os

from lightgbm import LGBMRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split

# ==========================================
# 1. FILE PATHS
# ==========================================
file1 = r"D:/Research/MATLAB/Daubechies/Precipitation/Monthly/Decomposed Data/Combined_rx5day_MON/MODWT_Level_7/MODWT_combined/Level_7.xlsx"

enso_file = r"D:/Research/MATLAB/Daubechies/Precipitation/Monthly/Nino_3_3.4_4_(HadISST)_from 1982.xlsx"

# Output folder
output_folder = r"D:/Research/MATLAB/Daubechies/Precipitation/Monthly/Target Nino/All stations as X/Single Input for X/Combined_rx5day_MON/LGBM/Nino_3.4/Level 7"

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================
df1 = pd.read_excel(file1)

df_enso = pd.read_excel(enso_file)

# ==========================================
# 3. CONCAT FEATURES (X) & TARGET (Y)
# ==========================================
X = df1

# Save combined X
X.to_excel(
    os.path.join(output_folder, "LGBM_Combined_all_X_data.xlsx"),
    index=False
)

# Target
Y = df_enso["nino_3.4"]

# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================
X = X.replace([np.inf, -np.inf], np.nan)

X = X.fillna(X.mean())

Y = Y.replace([np.inf, -np.inf], np.nan)

Y = Y.fillna(Y.mean())

# ========================================== # 5. TRAIN-TEST SPLIT # ========================================== 
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size=0.3, random_state=42 )



# ==========================================
# 6. FEATURE SCALING
# ==========================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# ==========================================
# 7. LGBM MODEL
# ==========================================
model = LGBMRegressor(

    # Core boosting
    boosting_type='gbdt',
    objective='regression',

    # Tree parameters
    n_estimators=3000,
    learning_rate=0.005,
    max_depth=8,
    num_leaves=63,

    # Data sampling
    subsample=0.8,
    subsample_freq=1,

    # Feature sampling
    colsample_bytree=0.8,

    # Minimum child conditions
    min_child_samples=10,
    min_child_weight=1e-3,

    # Regularization
    reg_alpha=0.5,
    reg_lambda=1.0,

    # Split gain
    min_split_gain=0.01,

    # Histogram binning
    max_bin=255,

    # Extra randomness
    extra_trees=False,

    # Parallel processing
    n_jobs=-1,

    # Randomness
    random_state=42,

    # Verbosity
    verbosity=-1
)

# ==========================================
# 8. TRAIN MODEL
# ==========================================
model.fit(
    X_train_scaled,
    y_train
)

# ==========================================
# 9. PREDICTIONS
# ==========================================
y_train_pred = model.predict(X_train_scaled)

y_test_pred = model.predict(X_test_scaled)

# ==========================================
# 10. TRAINING METRICS
# ==========================================
train_mae = mean_absolute_error(
    y_train,
    y_train_pred
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        y_train_pred
    )
)

train_r2 = r2_score(
    y_train,
    y_train_pred
)

train_r, _ = pearsonr(
    y_train,
    y_train_pred
)

train_r2_pearson = train_r ** 2

# ==========================================
# 11. TESTING METRICS
# ==========================================
test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)

test_r, _ = pearsonr(
    y_test,
    y_test_pred
)

test_r2_pearson = test_r ** 2

# ==========================================
# 12. PRINT RESULTS
# ==========================================
print("\n===== TRAINING PERFORMANCE =====")

print(f"Train MAE          : {train_mae:.4f}")
print(f"Train RMSE         : {train_rmse:.4f}")
print(f"Train R2 (sklearn) : {train_r2:.4f}")
print(f"Train Pearson r    : {train_r:.4f}")
print(f"Train Pearson r^2  : {train_r2_pearson:.4f}")

print("\n===== TESTING PERFORMANCE =====")

print(f"Test MAE           : {test_mae:.4f}")
print(f"Test RMSE          : {test_rmse:.4f}")
print(f"Test R2 (sklearn)  : {test_r2:.4f}")
print(f"Test Pearson r     : {test_r:.4f}")
print(f"Test Pearson r^2   : {test_r2_pearson:.4f}")

# ==========================================
# 13. SAVE METRICS
# ==========================================
metrics_df = pd.DataFrame({

    "Dataset": ["Training", "Testing"],

    "MAE": [
        train_mae,
        test_mae
    ],

    "RMSE": [
        train_rmse,
        test_rmse
    ],

    "R2_sklearn": [
        train_r2,
        test_r2
    ],

    "Pearson_r": [
        train_r,
        test_r
    ],

    "Pearson_r2": [
        train_r2_pearson,
        test_r2_pearson
    ]
})

metrics_df.to_excel(
    os.path.join(
        output_folder,
        "LGBM_Model_Metrics.xlsx"
    ),
    index=False
)

# ==========================================
# 14. SAVE TRAIN PREDICTIONS
# ==========================================
train_predictions_df = pd.DataFrame({

    "y_train": y_train.values,

    "y_train_pred": y_train_pred
})

train_predictions_df.to_excel(
    os.path.join(
        output_folder,
        "LGBM_Training_Predictions.xlsx"
    ),
    index=False
)

# ==========================================
# 15. SAVE TEST PREDICTIONS
# ==========================================
test_predictions_df = pd.DataFrame({

    "y_test": y_test.values,

    "y_test_pred": y_test_pred
})

test_predictions_df.to_excel(
    os.path.join(
        output_folder,
        "LGBM_Testing_Predictions.xlsx"
    ),
    index=False
)

# ==========================================
# 16. FEATURE IMPORTANCE
# ==========================================
importance_df = pd.DataFrame({

    "Feature": X.columns,

    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

importance_df.to_excel(
    os.path.join(
        output_folder,
        "LGBM_Feature_Importance.xlsx"
    ),
    index=False
)

print("\nAll outputs saved successfully.")