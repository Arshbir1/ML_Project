# --- PHASE 1: SETUP AND LIBRARIES ---
import pandas as pd
import numpy as np
import warnings
from scipy.stats import skew

# Models
from sklearn.linear_model import LassoCV, RidgeCV

# NOTE: We are intentionally NOT using a scaler.
# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- PHASE 2: DATA LOADING AND PREPARATION ---
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print(" Datasets loaded successfully.")
except FileNotFoundError:
    print(" Error: train.csv or test.csv not found.")
    exit()

# Store Test IDs for submission
test_ids = df_test['Id']
df_train = df_train.drop('Id', axis=1)
df_test = df_test.drop('Id', axis=1)

# --- PHASE 3: AGGRESSIVE PREPROCESSING & FEATURE ENGINEERING ---
print(" Starting AGGRESSIVE preprocessing...")

# Remove a few extreme outliers that can skew linear models
df_train = df_train.drop(df_train[(df_train['UsableArea'] > 4500) & (df_train['HotelValue'] < 300000)].index)

# Log-transform the target variable (this is essential)
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])
y_train = df_train['HotelValue']

# TARGET ENCODING: Create a powerful location-based feature
# We do this before dropping the target from the main dataframe
district_map = df_train.groupby('District')['HotelValue'].median()
df_train['DistrictValue'] = df_train['District'].map(district_map)
df_test['DistrictValue'] = df_test['District'].map(district_map)
# Fill any districts in the test set that weren't in the train set
df_test['DistrictValue'] = df_test['DistrictValue'].fillna(y_train.median())

# Now we can drop the target for combining
df_train = df_train.drop('HotelValue', axis=1)
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

# --- FEATURE CREATION ---
print("  - Engineering a large set of new features...")

# 1. Base Engineered Features
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBath'] = all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) + all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths'])
all_data['PropertyAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['YearsSinceRemod'] = all_data['YearSold'] - all_data['RenovationYear']

# 2. Interaction & Polynomial Features
all_data['OverallGrade'] = all_data['OverallQuality'] * all_data['OverallCondition']
all_data['TotalSF_x_OverallGrade'] = all_data['TotalSF'] * all_data['OverallGrade']
for col in ['OverallQuality', 'UsableArea', 'TotalSF', 'PropertyAge', 'DistrictValue']:
    all_data[f'{col}-s2'] = all_data[col] ** 2 # Squared term
    all_data[f'{col}-s3'] = all_data[col] ** 3 # Cubed term

# --- ROBUST IMPUTATION ---
# Impute categorical NaNs with 'None'
for col in ('ServiceLaneType', 'BasementHeight', 'BasementCondition', 'BasementExposure', 'BasementFacilityType1', 'BasementFacilityType2', 'BoundaryFence', 'ExtraFacility', 'ParkingType', 'ParkingFinish', 'ParkingQuality', 'ParkingCondition', 'LoungeQuality', 'FacadeType'):
    all_data[col] = all_data[col].fillna('None')

# Impute numericals using the median of the neighborhood where applicable
all_data['RoadAccessLength'] = all_data.groupby('District')['RoadAccessLength'].transform(lambda x: x.fillna(x.median()))

# Catch-all for any remaining NaNs
numeric_cols = all_data.select_dtypes(include=np.number).columns
all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].median())
categorical_cols = all_data.select_dtypes(include='object').columns
all_data[categorical_cols] = all_data[categorical_cols].fillna('None')

# Log-transform skewed numerical features
numeric_feats = all_data.select_dtypes(include=np.number).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.75].index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# One-Hot Encode all categorical features
all_data = pd.get_dummies(all_data, drop_first=True)

# Separate back into training and test sets
X = all_data[:len(y_train)]
X_test = all_data[len(y_train):]

print(" Preprocessing complete. NO SCALING APPLIED.")

# --- PHASE 4: MODEL TRAINING WITH AUTOMATIC TUNING ---
print(" Training and tuning the final model ensemble...")

# 1. LASSO MODEL
# Use LassoCV to automatically find the best alpha
lasso_model = LassoCV(alphas=[0.0001, 0.0003, 0.0005, 0.001], random_state=42, cv=10, max_iter=10000)
lasso_model.fit(X, y_train)
print(f"  - Best Lasso alpha found: {lasso_model.alpha_:.4f}")

# 2. RIDGE MODEL
# Use RidgeCV to automatically find the best alpha
ridge_model = RidgeCV(alphas=[14, 16, 18, 20, 22], cv=10)
ridge_model.fit(X, y_train)
print(f"  - Best Ridge alpha found: {ridge_model.alpha_:.1f}")

# --- PHASE 5: BLENDING AND SUBMISSION ---
print(" Blending predictions and creating submission file...")

# Predict using both tuned models
lasso_log_preds = lasso_model.predict(X_test)
ridge_log_preds = ridge_model.predict(X_test)

# A simple 50/50 blend of two strong models is very robust
blended_log_preds = 0.5 * lasso_log_preds + 0.5 * ridge_log_preds

# Reverse the log transformation to get the final prices
final_predictions = np.expm1(blended_log_preds)

# Create the submission file
submission = pd.DataFrame({'Id': test_ids, 'HotelValue': final_predictions})
submission.to_csv('submission_aggressive_no_scaling.csv', index=False)

print("\n Submission file 'submission_aggressive_no_scaling.csv' created successfully!")
print("This is your most aggressively preprocessed attempt without scaling. Good luck.")
print(submission.head())
