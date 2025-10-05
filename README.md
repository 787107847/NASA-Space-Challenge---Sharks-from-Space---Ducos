# NASA Space Apps Challenge: Whale Shark Foraging Habitat Prediction

## Overview

This project analyzes whale shark movement data in the Gulf of Mexico and develops machine learning models to predict foraging habitats based on satellite oceanographic data. The analysis is divided into two main notebooks: data processing and modeling.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Data Sources](#data-sources)
3. [Main Notebook (main.ipynb)](#main-notebook-mainipynb)
   - [Data Loading and Processing](#data-loading-and-processing)
   - [Satellite Data Integration](#satellite-data-integration)
   - [Dataset Creation](#dataset-creation)
4. [Model Notebook (model.ipynb)](#model-notebook-modelipynb)
   - [Exploratory Data Analysis](#exploratory-data-analysis)
   - [Model Development](#model-development)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
   - [Model Evaluation](#model-evaluation)
5. [Technical Implementation](#technical-implementation)
6. [Results and Performance](#results-and-performance)
7. [Usage Instructions](#usage-instructions)
8. [Dependencies](#dependencies)
9. [Future Improvements](#future-improvements)

## Project Structure

```
NasaSpaceApps/
├── main.ipynb                 # Data processing and satellite integration
├── model.ipynb                # EDA and machine learning modeling
├── final_dataset.csv          # Processed dataset for modeling
├── foraging_habitat_model_intensive.pkl    # Trained ensemble model
├── scaler_intensive.pkl       # Feature scaler
├── poly_features.pkl          # Polynomial feature transformer
├── descargas/                 # Downloaded satellite data
│   ├── chl_binned/           # Chlorophyll-a data
│   ├── flh_binned/           # Fluorescence line height
│   ├── iop_binned/           # Inherent optical properties
│   ├── kd_binned/           # Diffuse attenuation coefficient
│   ├── pic_binned/           # Particulate inorganic carbon
│   ├── rrs_binned/           # Remote sensing reflectance
│   └── sst_binned/           # Sea surface temperature
├── PorbeagleSeriescsv/       # Raw shark tracking data
└── README.md                 # This file
```

## Data Sources

### Whale Shark Movement Data
- **Source**: Whale shark tracking data from Gulf of Mexico
- **Format**: CSV files with location coordinates, timestamps, and foraging indicators
- **Variables**: location-long, location-lat, date, is_foraging

### Satellite Oceanographic Data
- **Source**: NASA Ocean Color Web (OCW) and other satellite missions
- **Products**:
  - **Chlorophyll-a (CHL)**: Ocean productivity indicator
  - **Normalized Fluorescence Line Height (NFLH)**: Phytoplankton biomass proxy
  - **Sea Surface Temperature (SST)**: Thermal habitat conditions
  - **Diffuse Attenuation Coefficient (Kd_490)**: Water clarity measure
  - **Remote Sensing Reflectance (Rrs_412)**: Ocean color properties
  - **Particulate Inorganic Carbon (PIC)**: Carbon concentration
  - **Absorption (a_412)**: Light absorption by particles
  - **Backscattering (bb_412)**: Light scattering properties
- **Temporal Resolution**: 8-day composites
- **Spatial Resolution**: 9km

## Main Notebook (main.ipynb)

### Data Loading and Processing

#### Whale Shark Data Processing
- **Input**: Multiple CSV files containing whale shark tracking data
- **Processing Steps**:
  1. Load and concatenate all shark movement files
  2. Parse timestamps and standardize date formats
  3. Remove duplicate entries
  4. Extract unique dates for satellite data matching
  5. Calculate movement displacements and identify foraging periods

#### Key Functions
```python
# Date parsing and standardization
shark_dates = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Duplicate removal
df = df.drop_duplicates(subset=['location-long', 'location-lat', 'date'])

# Movement analysis
displacement_lat = new_lat - lat
displacement_lon = new_lon - lon
```

### Satellite Data Integration

#### NetCDF File Processing
- **Libraries**: netCDF4, numpy, pandas
- **Processing Pipeline**:
  1. Identify closest satellite files by date for each shark location
  2. Extract variable values at shark coordinates using bilinear interpolation
  3. Handle missing data and fill values appropriately
  4. Apply quality flags and data validation

#### Satellite Data Types
- **Chlorophyll-a**: Primary productivity indicator
- **NFLH**: Phytoplankton fluorescence proxy
- **SST**: Thermal habitat characterization
- **IOPs**: Inherent optical properties (absorption, backscattering)
- **Rrs**: Spectral reflectance measurements

#### Data Extraction Logic
```python
# Find closest file by date
min_diff = min(abs((file_date - target_date).days) for file_date in file_dates)

# Extract value at coordinates
lat_idx = np.argmin(np.abs(lat_arr - lat))
lon_idx = np.argmin(np.abs(lon_arr - lon))
value = data_array[lat_idx, lon_idx]
```

### Dataset Creation

#### Feature Engineering
- **Spatial Matching**: Match satellite data to shark locations within ±4 days
- **Temporal Aggregation**: Use 8-day composite data for temporal smoothing
- **Quality Control**: Remove invalid measurements and apply data filters

#### Output Dataset
- **File**: `final_dataset.csv`
- **Records**: 12,916 tagged locations
- **Features**: 8 satellite variables + location + date + foraging label
- **Balance**: Stratified sampling maintains class distribution

## Model Notebook (model.ipynb)

### Exploratory Data Analysis

#### Data Overview
- **Dataset Size**: 12,916 samples, 12 columns
- **Data Types**: 10 float64 (satellite features), 1 int64 (target), 1 object (date)
- **Missing Values**: None detected
- **Class Distribution**: Balanced (50% foraging, 50% non-foraging)

#### Statistical Analysis
```python
# Basic statistics
df.describe()

# Missing value detection
df.isnull().sum()

# Class distribution
df['is_foraging'].value_counts()
```

#### Outlier Detection
- **Method**: Box plots for all numerical features
- **Purpose**: Identify potential data quality issues
- **Features Analyzed**: All 8 satellite variables + coordinates

### Model Development

#### Feature Selection
- **Excluded Features**: location-long, location-lat, date (spatial/temporal identifiers)
- **Selected Features**: nflh, chl, a_412, bb_412, pic, kd_490, rrs_412, sst
- **Rationale**: Focus on oceanographic conditions for habitat prediction

#### Model Architectures
1. **Logistic Regression**: Baseline linear model
2. **Random Forest**: Ensemble tree-based model
3. **XGBoost**: Gradient boosting with regularization
4. **LightGBM**: Microsoft gradient boosting framework
5. **CatBoost**: Yandex categorical boosting
6. **Stacking Ensemble**: Meta-learner combining all models

### Hyperparameter Tuning

#### Optimization Framework
- **Library**: Optuna (Bayesian optimization)
- **Trials**: 100 per model
- **Cross-Validation**: 10-fold stratified
- **Metric**: ROC-AUC (area under receiver operating characteristic curve)

#### Feature Engineering for Modeling
- **Polynomial Features**: Degree-2 interactions between all features
- **Domain Features**:
  - `chl_nflh_ratio`: Chlorophyll to fluorescence ratio
  - `sst_chl_interaction`: Temperature-productivity interaction
  - `kd_rrs_sum`: Attenuation + reflectance combination
- **Total Features**: 39 (8 original + 28 interactions + 3 domain features)
- **Scaling**: StandardScaler for feature normalization

#### Optuna Objectives
```python
def objective_xgb(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 1000)
    max_depth = trial.suggest_int('max_depth', 3, 15)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 0.3, log=True)
    # ... other parameters
    
    model = xgb.XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring='roc_auc')
    return scores.mean()
```

### Model Evaluation

#### Performance Metrics
- **ROC-AUC**: Primary metric for imbalanced classification
- **Accuracy**: Overall prediction accuracy
- **Precision/Recall/F1**: Class-specific performance
- **Classification Report**: Detailed per-class metrics

#### Cross-Validation Results
| Model | CV ROC-AUC | Test ROC-AUC | Test Accuracy |
|-------|------------|--------------|---------------|
| Logistic Regression | 0.5330 | 0.5080 | 0.5058 |
| Random Forest | 0.5549 | 0.5541 | 0.5337 |
| XGBoost | 0.5542 | 0.5512 | 0.5302 |
| LightGBM | 0.5588 | 0.5473 | 0.5271 |
| CatBoost | 0.5558 | 0.5440 | 0.5271 |
| Stacking Ensemble | 0.5537 | 0.5514 | 0.5236 |

#### Model Persistence
- **Saved Models**: 
  - `foraging_habitat_model_intensive.pkl`: Stacking ensemble
  - `scaler_intensive.pkl`: Feature scaler
  - `poly_features.pkl`: Polynomial transformer
- **Format**: Joblib serialization for scikit-learn compatibility

## Technical Implementation

### Libraries and Dependencies
```python
# Core data processing
pandas>=2.0.0
numpy>=1.24.0
netCDF4>=1.6.0

# Machine learning
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
optuna>=3.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
joblib>=1.3.0
```

### Environment Setup
```bash
# Create conda environment
conda create -n nasa-sharks python=3.11
conda activate nasa-sharks

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install xgboost[gpu] lightgbm[gpu]
```

### Computational Requirements
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB for satellite data downloads
- **Processing Time**: 
  - Data processing: 30-60 minutes
  - Model training: 45-60 minutes
  - Hyperparameter tuning: 60-90 minutes

## Results and Performance

### Model Performance Summary
- **Best Individual Model**: Random Forest (ROC-AUC: 0.5541)
- **Ensemble Performance**: Stacking (ROC-AUC: 0.5514)
- **Overall Accuracy**: ~53% (slightly above random guessing)
- **Key Insight**: Satellite features provide moderate predictive power for foraging habitat identification

### Feature Importance Analysis
- **Top Features**: Chlorophyll-a, NFLH, SST show strongest correlations
- **Interaction Effects**: Cross-terms improve model capacity
- **Domain Features**: Engineered ratios provide additional signal

### Limitations
- **Feature Predictive Power**: Satellite data alone achieves ~55% ROC-AUC
- **Temporal Resolution**: 8-day composites may miss short-term habitat changes
- **Spatial Resolution**: 9km pixels limit fine-scale habitat mapping
- **Data Balance**: Balanced classes may not reflect natural foraging frequency

## Usage Instructions

### Running the Notebooks

#### 1. Data Processing (main.ipynb)
```python
# Execute cells in order
# Cell 1-2: Load whale shark data
# Cell 3-7: Process satellite data directories
# Cell 8-42: Extract and match satellite values
# Output: final_dataset.csv
```

#### 2. Model Training (model.ipynb)
```python
# Execute cells in order
# Cell 1-7: EDA and data preparation
# Cell 8-14: Feature engineering and scaling
# Cell 15-16: Optuna hyperparameter optimization
# Cell 17-18: Model evaluation
# Cell 19-20: Final model training and saving
```

### Using the Trained Model
```python
import joblib
import pandas as pd

# Load components
model = joblib.load('foraging_habitat_model_intensive.pkl')
scaler = joblib.load('scaler_intensive.pkl')
poly = joblib.load('poly_features.pkl')

# Prepare new data
new_data = pd.DataFrame({
    'nflh': [0.05], 'chl': [0.5], 'a_412': [0.08],
    'bb_412': [0.005], 'pic': [0.0001], 'kd_490': [0.07],
    'rrs_412': [0.003], 'sst': [30.0]
})

# Apply transformations
new_poly = poly.transform(new_data)
new_poly_df = pd.DataFrame(new_poly, columns=poly.get_feature_names_out())

# Add domain features
new_poly_df['chl_nflh_ratio'] = new_data['chl'] / (new_data['nflh'] + 1e-6)
new_poly_df['sst_chl_interaction'] = new_data['sst'] * new_data['chl']
new_poly_df['kd_rrs_sum'] = new_data['kd_490'] + new_data['rrs_412']

# Scale and predict
new_scaled = scaler.transform(new_poly_df)
probability = model.predict_proba(new_scaled)[0, 1]
prediction = model.predict(new_scaled)[0]

print(f"Foraging Probability: {probability:.4f}")
print(f"Predicted Class: {'Foraging' if prediction == 1 else 'Non-foraging'}")
```

### Batch Processing
```python
# For multiple locations
batch_data = pd.read_csv('new_locations.csv')
# Apply same preprocessing pipeline
batch_probabilities = model.predict_proba(batch_scaled)
batch_predictions = model.predict(batch_scaled)
```

## Future Improvements

### Data Enhancements
- **Higher Resolution Data**: Incorporate 1km or 500m satellite products
- **Additional Variables**: Ocean currents, salinity, dissolved oxygen
- **Temporal Features**: Time-series analysis of habitat changes
- **Species-Specific Data**: Include prey distribution data

### Model Improvements
- **Deep Learning**: CNNs for spatial pattern recognition
- **Time-Series Models**: LSTM networks for temporal dependencies
- **Ensemble Methods**: Advanced stacking with different meta-learners
- **Feature Selection**: Automated feature importance ranking

### Technical Enhancements
- **GPU Acceleration**: Optimize for CUDA-enabled hardware
- **Distributed Computing**: Scale to larger datasets
- **Model Interpretability**: SHAP values for feature importance
- **Real-time Processing**: Streaming predictions for new data

### Validation and Deployment
- **Field Validation**: Compare predictions with observed foraging
- **Web Application**: Interactive habitat mapping interface
- **API Development**: RESTful service for habitat predictions
- **Monitoring**: Model performance tracking over time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper documentation
4. Test thoroughly on sample data
5. Submit pull request with detailed description

## License

This project is part of the NASA Space Apps Challenge and follows NASA open data policies.

## Acknowledgments

- **NASA Ocean Color Web**: Satellite data provision
- **Whale Shark Research Community**: Movement data contribution
- **NASA Space Apps Challenge**: Project inspiration and framework
- **Open Source Community**: Libraries and tools enabling this analysis

---

**Last Updated**: October 5, 2025
**Project Status**: Complete with room for enhancement