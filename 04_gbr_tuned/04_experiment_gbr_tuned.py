# --- PHASE 1: SETUP AND DATA LOADING ---
import pandas as pd
import numpy as np
from scipy.stats import skew
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Load Data
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("Training and test datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure train.csv and test.csv are in the same directory.")
    exit()

# Store IDs for the final submission file
train_ids = df_train['Id']
test_ids = df_test['Id']
df_train = df_train.drop('Id', axis=1)
df_test = df_test.drop('Id', axis=1)

print(f"Initial Training data shape: {df_train.shape}")


# --- PHASE 2: PREPROCESSING AND FEATURE ENGINEERING (OPTIMIZED FOR GBR) ---

# 2.1 Handle Outliers
df_train = df_train.drop(df_train[(df_train['UsableArea']>4500) & (df_train['HotelValue']<300000)].index)
print(f"Training data shape after outlier removal: {df_train.shape}")

# 2.2 Log Transform Target Variable (still very important!)
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])
y_train = df_train['HotelValue']
df_train = df_train.drop('HotelValue', axis=1)

# 2.3 Combine Data for Consistent Preprocessing
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
print(f"Combined data shape: {all_data.shape}")

# 2.4 Feature Engineering (Creating more predictive features)
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBath'] = all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) + all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths'])
all_data['WasRemod'] = (all_data['ConstructionYear'] != all_data['RenovationYear']).astype(int)
all_data['PropertyAge'] = all_data['YearSold'] - all_data['ConstructionYear']

# 2.5 Impute Missing Values (Filling NaNs)
# Categorical
for col in ['ServiceLaneType', 'BasementHeight', 'BasementCondition', 'BasementExposure', 'BasementFacilityType1', 'BasementFacilityType2', 'BoundaryFence', 'ExtraFacility', 'ParkingType', 'ParkingFinish', 'ParkingQuality', 'ParkingCondition', 'LoungeQuality', 'FacadeType', 'ElectricalSystem']:
    all_data[col] = all_data[col].fillna('None')

# Numerical
for col in ['BasementFacilitySF1', 'BasementFacilitySF2', 'BasementUnfinishedSF','BasementTotalSF', 'BasementFullBaths', 'BasementHalfBaths', 'ParkingCapacity', 'ParkingArea', 'FacadeArea']:
    all_data[col] = all_data[col].fillna(0)

# Use median of the neighborhood to fill RoadAccessLength
all_data['RoadAccessLength'] = all_data.groupby('District')['RoadAccessLength'].transform(lambda x: x.fillna(x.median()))

# *** THIS IS THE FIX: A catch-all for any remaining numerical NaNs ***
all_data = all_data.fillna(all_data.select_dtypes(include=np.number).median())

# 2.6 One-Hot Encode Categorical Features
all_data = pd.get_dummies(all_data, drop_first=True) # drop_first reduces dimensionality

# 2.7 Separate back into Training and Test sets
X = all_data[:len(y_train)]
X_test = all_data[len(y_train):]

# Sanity check for remaining NaNs
if X.isnull().sum().sum() > 0:
    print("Warning: Missing values still exist in the training data after preprocessing!")
else:
    print("Preprocessing complete. No missing values found in training data.")


# --- PHASE 3: AGGRESSIVE HYPERPARAMETER TUNING ---
# (Rest of the script remains the same)

print("\n--- Tuning the GradientBoostingRegressor ---")
print("This will take several minutes...")

# Define a more aggressive parameter grid to search
param_grid = {
    'n_estimators': [500, 1000, 1500],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [3, 4],
    'subsample': [0.7, 0.8], # Adds stochasticity
    'max_features': ['sqrt'], # Reduces variance
    'min_samples_leaf': [15, 20] # Prevents overfitting to noisy points
}

# Setup GridSearchCV
gs = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42, loss='huber'), # Huber loss is robust to outliers
    param_grid=param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    verbose=2, # Shows progress
    n_jobs=-1 # Uses all available CPU cores
)

# Fit the grid search to find the best model
gs.fit(X, y_train)
best_gbr = gs.best_estimator_

print("\nGridSearchCV Complete.")
print(f"Best Cross-Validation Score (RMSLE): {-gs.best_score_:.5f}")
print(f"Best GBR Parameters Found: {gs.best_params_}")


# --- PHASE 4: FINAL PREDICTION AND SUBMISSION ---

print("\n--- Generating Final Submission File ---")

# The best_gbr model is already trained on the full training data by GridSearchCV
log_predictions = best_gbr.predict(X_test)

# Reverse the log transformation
final_predictions = np.expm1(log_predictions)

# Create the submission dataframe
submission = pd.DataFrame({'Id': test_ids, 'HotelValue': final_predictions})
submission.to_csv('submission_champion_gbr.csv', index=False)

print("\nSubmission file 'submission_champion_gbr.csv' created successfully!")
print("This file contains predictions from the best-tuned Gradient Boosting model.")
print(submission.head())
