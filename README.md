# Bangladesh ENSO Teleconnection

This repository contains the computational workflow for my research investigating the relationship between Bangladesh hydroclimatic variability and ENSO-related tropical Pacific SST variability.

The study uses monthly climate data from 29 meteorological stations (1982–2022) and examines Niño 1+2, Niño 3, and Niño 3.4 using Pearson correlation, wavelet analysis, and machine-learning regression.

## Repository Contents

### `Correlation/`

Contains the Pearson correlation and lagged-correlation analysis used to investigate statistical relationships between Bangladesh climate variables and ENSO indices.

### `Original_Data_Scripts/`

Contains machine-learning regression scripts using the original monthly climate signals without wavelet decomposition.

The experiments are organized by:

- **`Multi_Station/`** — climate information from all 29 stations is used together.
- **`Single_Station/`** — each station is modeled separately.
- **`Single_Index/`** — Tmax, Tmin, and individual precipitation indices are modeled separately.
- **`Composite_Indices/`** — the five precipitation indices are combined as predictors.

### `Wavelet_Components/`

Contains the scripts required for wavelet-enhanced machine-learning regression.

The same scripts process the required MATLAB data for the different climate variables and decomposition settings, so separate folders for individual indices or wavelet methods are not required.

The wavelet approaches include:

- Haar
- Db4
- Coif3
- TQWT

## Machine-Learning Models

The regression framework uses:

- Random Forest (RF)
- XGBoost
- LightGBM

The objective is to evaluate whether wavelet-based multi-resolution representations of Bangladesh hydroclimatic variability provide additional information for characterizing ENSO-related SST variability.

This repository provides the computational record of the analysis supporting the associated research.

## Contact

**James Sarker Shuva**  
Civil Engineering  
Leading University, Sylhet, Bangladesh

Research interests include:

- Hydroclimatology
- Climate variability and teleconnections
- Hydrology
- Machine learning
- Wavelet analysis
- Extreme precipitation
- GIS and spatial analysis

**GitHub:** `sarkerjams-coder`
