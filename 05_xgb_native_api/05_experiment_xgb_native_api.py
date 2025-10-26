# --- PHASE 1: SETUP AND DATA LOADING ---
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# Load Data
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("✅ Training and test datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure train.csv and test.csv are in the same directory.")
    exit()

# Store IDs for the final submission file
test_ids = df_test['Id']
# Drop Id columns as they are not features
df_train = df_train.drop('Id', axis=1)
df_test = df_test.drop('Id', axis=1)

print(f"Initial Training data shape: {df_train.shape}")


# --- PHASE 2: ADVANCED PREPROCESSING & FEATURE ENGINEERING ---

# 2.1 Handle Outliers (a gentle approach)
df_train = df_train.drop(df_train[(df_train['UsableArea']>4000) & (df_train['HotelValue']<300000)].index)
print(f"Training data shape after outlier removal: {df_train.shape}")

# 2.2 Log Transform Target Variable (crucial for skewed targets)
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])
y_train = df_train['HotelValue']
df_train = df_train.drop('HotelValue', axis=1)

# 2.3 Combine Data for Consistent Preprocessing
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)
print(f"Combined data shape: {all_data.shape}")

# 2.4 Feature Engineering
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBath'] = all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) + all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths'])
all_data['PropertyAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['YearsSinceRemod'] = all_data['YearSold'] - all_data['RenovationYear']
# NEW: Interaction term for overall quality and condition
all_data['OverallGrade'] = all_data['OverallQuality'] * all_data['OverallCondition']

# 2.5 Impute Missing Values
# Categorical
for col in ('ServiceLaneType', 'BasementHeight', 'BasementCondition', 'BasementExposure', 'BasementFacilityType1', 'BasementFacilityType2', 'BoundaryFence', 'ExtraFacility', 'ParkingType', 'ParkingFinish', 'ParkingQuality', 'ParkingCondition', 'LoungeQuality', 'FacadeType', 'ElectricalSystem'):
    all_data[col] = all_data[col].fillna('None')
# Numerical
for col in ('ParkingCapacity', 'ParkingArea', 'FacadeArea', 'BasementFullBaths', 'BasementHalfBaths', 'BasementTotalSF'):
     all_data[col] = all_data[col].fillna(0)
# Use median of the neighborhood to fill RoadAccessLength
all_data['RoadAccessLength'] = all_data.groupby('District')['RoadAccessLength'].transform(lambda x: x.fillna(x.median()))

# 2.6 One-Hot Encode Categorical Features
all_data = pd.get_dummies(all_data, drop_first=True)

# 2.7 Final Catch-All Imputation
all_data = all_data.fillna(all_data.median())

# 2.8 Separate back into Training and Test sets
X = all_data[:len(y_train)]
X_test = all_data[len(y_train):]
print("✅ Preprocessing and feature engineering complete.")


# --- PHASE 3: XGBOOST MODEL TRAINING AND TUNING ---

print("\n--- Training XGBoost Model with Cross-Validation ---")

# We will use xgb.cv to find the best number of boosting rounds (trees)
# This is more efficient than GridSearchCV for this specific parameter.
dtrain = xgb.DMatrix(X, label=y_train)
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.03, # learning_rate
    'max_depth': 4,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'seed': 42
}

# Run cross-validation
# This will train 10 models and stop when the validation score hasn't improved for 50 rounds
cv_results = xgb.cv(
    params=params,
    dtrain=dtrain,
    num_boost_round=2000,
    nfold=10,
    early_stopping_rounds=50,
    verbose_eval=100,
    as_pandas=True
)

best_nrounds = cv_results.shape[0]
print(f"Optimal number of boosting rounds: {best_nrounds}")
best_rmse = cv_results['test-rmse-mean'].iloc[-1]
print(f"Best CV Score (RMSLE): {best_rmse:.5f}")


# --- PHASE 4: FINAL PREDICTION AND SUBMISSION ---

print("\n--- Generating Final Submission File ---")

# Train the final model on ALL training data with the optimal number of rounds
final_xgb = xgb.train(params, dtrain, num_boost_round=best_nrounds)

# Create DMatrix for the test set
dtest = xgb.DMatrix(X_test)

# Make predictions
log_predictions = final_xgb.predict(dtest)

# Reverse the log transformation
final_predictions = np.expm1(log_predictions)

# Create the submission dataframe
submission = pd.DataFrame({'Id': test_ids, 'HotelValue': final_predictions})
submission.to_csv('submission_xgboost.csv', index=False)

print("\n🚀 Submission file 'submission_xgboost.csv' created successfully!")
print("This file was generated using a tuned XGBoost model.")
print(submission.head())
