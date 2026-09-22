import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

input_file = "D:Research/KNN_T/Merged_tmax_data.csv"

df = pd.read_csv(input_file)

# Target column
col_index = 18

# Identify originally missing values
missing_mask = df.iloc[:, col_index].isna()

print(
    f"Missing values in '{df.columns[col_index]}' before imputation:",
    missing_mask.sum()
)

# KNN imputation
knn_imputer = KNNImputer(n_neighbors=5)

imputed_data = knn_imputer.fit_transform(df)

# Extract target column
imputed_values = imputed_data[:, col_index]

# Round to 2 decimal places
imputed_values = np.round(imputed_values, 2)

# Replace only originally missing target values
df.loc[missing_mask, df.columns[col_index]] = imputed_values[missing_mask]

# Save
output_file = "E:/Seminar/Tmax_imputed_KNN_Barisal.xlsx"
df.to_excel(output_file, index=False)

print("Imputation completed.")
print("Saved to:", output_file)