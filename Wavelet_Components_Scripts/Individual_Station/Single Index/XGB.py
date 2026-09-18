import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# ==========================================
# 1. FILE PATHS
# ==========================================
file1 = r"D:/Research/MATLAB/Daubechies/Precipitation/Monthly/Decomposed Data/Combined_rx5day_MON/MODWT_Level_7/MODWT_combined/Level_7.xlsx"

enso_file = r"D:/Research/MATLAB/Daubechies/Precipitation/Monthly/Nino_3_3.4_4_(HadISST)_from 1982.xlsx"

# Output folder
output_folder = r"D:/Research/MATLAB/Daubechies/Precipitation/Monthly/Target Nino/Individual_Station/Single Input for X/Combined_rx5day_MON/XGB/Nino_3.4/Level 7"

os.makedirs(output_folder, exist_ok=True)

# ==========================================
# 2. LOAD DATA
# ==========================================
df1 = pd.read_excel(file1)

df_enso = pd.read_excel(enso_file)

# ==========================================
# 3. DETECT STATIONS
# ==========================================
stations = sorted({

    col.split("_")[0]

    for col in df1.columns

})

Y = df_enso["nino_3.4"]

Y_clean = Y.replace(
    [np.inf, -np.inf],
    np.nan
)

Y_clean = Y_clean.fillna(
    Y_clean.mean()
)

all_metrics = []

for station in stations:

    print(f"\nProcessing station: {station}")

    station_folder = os.path.join(
        output_folder,
        station
    )

    os.makedirs(
        station_folder,
        exist_ok=True
    )

    X = df1[
        [c for c in df1.columns
         if c.startswith(station + "_")]
    ]

    X.to_excel(
        os.path.join(
            station_folder,
            "XGB_Combined_all_X_data.xlsx"
        ),
        index=False
    )

    # ==========================================
    # 4. HANDLE MISSING VALUES
    # ==========================================
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.mean())

    
    # ========================================== # 5. TRAIN-TEST SPLIT # ========================================== 
    X_train, X_test, y_train, y_test = train_test_split( X, Y_clean, test_size=0.3, random_state=42 )

    # ==========================================
    # 6. SCALING
    # ==========================================
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # 7. MODEL
    # ==========================================
    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=3,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.5,
        reg_lambda=1
    )

    # Train model
    model.fit(X_train_scaled, y_train)

    # ==========================================
    # 8. PREDICTION
    # ==========================================
    y_train_pred = model.predict(X_train_scaled)

    y_test_pred = model.predict(X_test_scaled)

    # ==========================================
    # 9. TRAINING METRICS
    # ==========================================
    train_mae = mean_absolute_error(y_train, y_train_pred)

    train_rmse = np.sqrt(
        mean_squared_error(y_train, y_train_pred)
    )

    train_r2 = r2_score(y_train, y_train_pred)

    train_r, _ = pearsonr(y_train, y_train_pred)

    train_r2_pearson = train_r ** 2

    # ==========================================
    # 10. TESTING METRICS
    # ==========================================
    test_mae = mean_absolute_error(y_test, y_test_pred)

    test_rmse = np.sqrt(
        mean_squared_error(y_test, y_test_pred)
    )

    test_r2 = r2_score(y_test, y_test_pred)

    test_r, _ = pearsonr(y_test, y_test_pred)

    test_r2_pearson = test_r ** 2

    # ==========================================
    # 11. PRINT RESULTS
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
    # 12. SAVE METRICS
    # ==========================================
    metrics_df = pd.DataFrame({

        "Dataset": ["Training", "Testing"],

        "MAE": [train_mae, test_mae],

        "RMSE": [train_rmse, test_rmse],

        "R2_sklearn": [train_r2, test_r2],

        "Pearson_r": [train_r, test_r],

        "Pearson_r2": [
            train_r2_pearson,
            test_r2_pearson
        ]
    })

    metrics_df.to_excel(
        os.path.join(station_folder, "Model_Metrics.xlsx"),
        index=False
    )
    
    all_metrics.append({

        "Station": station,
        
        "Train_MAE": train_mae,
        "Test_MAE": test_mae,
        
        "Train_RMSE": train_rmse,
        "Test_RMSE": test_rmse,
            
        "Train_R2": train_r2,
        "Test_R2": test_r2,
        
        "Train_Pearson_r2":
            train_r2_pearson,
            
        "Test_Pearson_r2":
            test_r2_pearson
    })

    # ==========================================
    # 13. SAVE TRAIN PREDICTIONS
    # ==========================================
    train_predictions_df = pd.DataFrame({

        "y_train": y_train.values,

        "y_train_pred": y_train_pred
    })

    train_predictions_df.to_excel(
        os.path.join(station_folder, "Training_Predictions.xlsx"),
        index=False
    )

    # ==========================================
    # 14. SAVE TEST PREDICTIONS
    # ==========================================
    test_predictions_df = pd.DataFrame({

        "y_test": y_test.values,

        "y_test_pred": y_test_pred
    })

    test_predictions_df.to_excel(
        os.path.join(station_folder, "Testing_Predictions.xlsx"),
        index=False
    )

    print(f"\n{station} completed successfully.")

summary_df = pd.DataFrame(
    all_metrics
)

summary_df.to_excel(
    os.path.join(
        output_folder,
        "XGB_All_Stations_Metrics.xlsx"
    ),
    index=False
)

print("\nAll stations processed successfully.")