
"""
Created on Mon May 25 17:50:52 2026

@author: Jams
"""

import pandas as pd
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr

# 1. FILE PATHS
file1 = r"D:/Research/Journal/LGBM, RF, XGB/Precipitation/Monthly/Combined_prcptot_MON.csv"
file2 = r"D:/Research/Journal/LGBM, RF, XGB/Precipitation/Monthly/Combined_r10mm_MON.csv"
file3 = r"D:/Research/Journal/LGBM, RF, XGB/Precipitation/Monthly/Combined_r20mm_MON.csv"
file4 = r"D:/Research/Journal/LGBM, RF, XGB/Precipitation/Monthly/Combined_rx1day_MON.csv"
file5 = r"D:/Research/Journal/LGBM, RF, XGB/Precipitation/Monthly/Combined_rx5day_MON.csv"

enso_file = r"D:/Research/Journal/LGBM, RF, XGB/Precipitation/Monthly/Nino_3_3.4_4_(HadISST)_from 1982.xlsx"

# Output folder
output_folder = r"D:/Research/Journal/LGBM, RF, XGB/Precipitation/Monthly/All stations as X/Multiple Input for X/RF/Nino_3.4"

os.makedirs(output_folder, exist_ok=True)

# 2. LOAD DATA
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)
df4 = pd.read_csv(file4)
df5 = pd.read_csv(file5)

df_enso = pd.read_excel(enso_file)

# 3. CONCAT FEATURES (X) & TARGET (Y)
X = pd.concat(
    [df1, df2, df3, df4, df5],
    axis=1
)

# Save combined X
X.to_excel(
    os.path.join(
        output_folder,
        "RF_Combined_all_X_data_raw.xlsx"
    ),
    index=False
)

# Target
Y = df_enso["nino_3.4"]


# 4. HANDLE MISSING VALUES
X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

X = X.fillna(X.mean())

Y = Y.replace(
    [np.inf, -np.inf],
    np.nan
)

Y = Y.fillna(Y.mean())


# 5. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.30,   # 30% test, 70% train
    random_state=42
)

# 6. SCALING
# (Optional for RF, but retained)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# 7. RANDOM FOREST MODEL
model = RandomForestRegressor(

    # Forest size
    n_estimators=2000,

    # Tree structure
    criterion='squared_error',
    max_depth=25,

    # Node splitting
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,

    # Feature selection
    max_features='sqrt',

    # Leaf control
    max_leaf_nodes=None,

    # Impurity reduction
    min_impurity_decrease=0.0,

    # Bootstrap sampling
    bootstrap=True,
    oob_score=True,

    # Parallel processing
    n_jobs=-1,

    # Reproducibility
    random_state=42,

    # Verbosity
    verbose=1,

    # Warm start
    warm_start=False,

    # Complexity pruning
    ccp_alpha=0.0,

    # Maximum samples
    max_samples=None
)

# 8. TRAIN MODEL
model.fit(
    X_train_scaled,
    y_train
)

# 9. PREDICTIONS
y_train_pred = model.predict(
    X_train_scaled
)

y_test_pred = model.predict(
    X_test_scaled
)

# 10. TRAINING METRICS
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

# 11. TESTING METRICS
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

# 12. PRINT RESULTS
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

# OOB score
print(f"\nOOB Score          : {model.oob_score_:.4f}")

# 13. SAVE METRICS
metrics_df = pd.DataFrame({

    "Dataset": [
        "Training",
        "Testing"
    ],

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
        "RF_Model_Metrics_raw.xlsx"
    ),
    index=False
)

# 14. SAVE TRAIN PREDICTIONS
train_predictions_df = pd.DataFrame({

    "y_train": y_train.values,

    "y_train_pred": y_train_pred
})

train_predictions_df.to_excel(
    os.path.join(
        output_folder,
        "RF_Training_Predictions.xlsx"
    ),
    index=False
)

# 15. SAVE TEST PREDICTIONS
test_predictions_df = pd.DataFrame({

    "y_test": y_test.values,

    "y_test_pred": y_test_pred
})

test_predictions_df.to_excel(
    os.path.join(
        output_folder,
        "RF_Testing_Predictions.xlsx"
    ),
    index=False
)

# 16. FEATURE IMPORTANCE
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
        "RF_Feature_Importance.xlsx"
    ),
    index=False
)

print("\nAll outputs saved successfully.")