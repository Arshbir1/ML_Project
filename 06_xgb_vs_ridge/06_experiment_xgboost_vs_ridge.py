import pandas as pd
import numpy as np
import xgboost as xgb
# matplotlib and seaborn imports removed
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# ---  PHASE 1: LOAD AND PREPROCESS DATA (ADVANCED) ---
print("--- Phase 1: Loading and Preprocessing Data (Advanced) ---")
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("Training and test datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}.")
    exit()

test_ids = df_test['Id']
df_train = df_train.drop('Id', axis=1)
df_test = df_test.drop('Id', axis=1)

y_train_log = np.log1p(df_train.pop('HotelValue'))
all_data = pd.concat([df_train, df_test], ignore_index=True)

# --- NEW STEP: FIXING DIRTY DATA (TA's HINT) ---
print("\n--- Applying TA's Hint: Correcting Logical Errors ---")

# 1. Find impossible RenovationYear dates (Renovated *after* being sold)
invalid_reno_count = all_data[all_data['RenovationYear'] > all_data['YearSold']].shape[0]
print(f"Found {invalid_reno_count} rows where RenovationYear > YearSold. Correcting them...")
# "Modify the data": Assume the renovation year is invalid, set to ConstructionYear
all_data.loc[all_data['RenovationYear'] > all_data['YearSold'], 'RenovationYear'] = all_data.loc[all_data['RenovationYear'] > all_data['YearSold'], 'ConstructionYear']

# 2. Find impossible ConstructionYear dates (Built *after* being sold)
invalid_build_count = all_data[all_data['ConstructionYear'] > all_data['YearSold']].shape[0]
print(f"Found {invalid_build_count} rows where ConstructionYear > YearSold. Correcting them...")
# "Modify the data": Set ConstructionYear to be the same as YearSold
all_data.loc[all_data['ConstructionYear'] > all_data['YearSold'], 'ConstructionYear'] = all_data.loc[all_data['ConstructionYear'] > all_data['YearSold'], 'YearSold']


# --- FEATURE ENGINEERING (PART 1: As before, but now with clean data) ---
all_data['HotelAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['YearsSinceRenovation'] = all_data['YearSold'] - all_data['RenovationYear']
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBathrooms'] = (all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) +
                               all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths']))

# --- NEW STEP: MANUAL ORDINAL MAPPING (Optimized for Tree Models) ---
print("\nApplying 'various preprocessing methods': Mapping ordinal features...")
# Map ordinal features to numbers. 'None' will be 0.
qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
all_data['ExteriorQuality'] = all_data['ExteriorQuality'].map(qual_map).fillna(3)
all_data['ExteriorCondition'] = all_data['ExteriorCondition'].map(qual_map).fillna(3)
all_data['BasementHeight'] = all_data['BasementHeight'].map(qual_map).fillna(0)
all_data['BasementCondition'] = all_data['BasementCondition'].map(qual_map).fillna(0)
all_data['HeatingQuality'] = all_data['HeatingQuality'].map(qual_map).fillna(3)
all_data['KitchenQuality'] = all_data['KitchenQuality'].map(qual_map).fillna(3)
all_data['LoungeQuality'] = all_data['LoungeQuality'].map(qual_map).fillna(0)
all_data['ParkingQuality'] = all_data['ParkingQuality'].map(qual_map).fillna(0)
all_data['ParkingCondition'] = all_data['ParkingCondition'].map(qual_map).fillna(0)
all_data['PoolQuality'] = all_data['PoolQuality'].map(qual_map).fillna(0)
all_data['BoundaryFence'] = all_data['BoundaryFence'].map({'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}).fillna(0)

# --- IMPUTATION (Fill remaining NaNs) ---
# Impute numerical features with median
for col in all_data.select_dtypes(include=np.number).columns:
    all_data[col] = all_data[col].fillna(all_data[col].median())
# Impute categorical features with mode
for col in all_data.select_dtypes(include='object').columns:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

# --- FEATURE ENGINEERING (PART 2: Interaction Features) ---
print("Engineering new interaction features...")
all_data['OverallQuality_x_TotalSF'] = all_data['OverallQuality'] * all_data['TotalSF']
all_data['HotelAge_x_OverallQuality'] = all_data['HotelAge'] * all_data['OverallQuality']

# --- FINAL ENCODING & SPLITTING ---
# One-hot encode only the *remaining* categorical columns
all_data = pd.get_dummies(all_data, drop_first=True)

# Separate back into train and test
X = all_data[:len(df_train)]
X_test = all_data[len(df_train):]
feature_names = X.columns # Variable is still needed for DataFrame creation, even if not plotted

# Create a scaled version for the Ridge model
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete. All features are now numerical.")

# --- PHASE 2: MODEL TRAINING & COMPARISON ---
print("\n--- Phase 2: Training and Tuning Models ---")
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Model 1: Re-train the Champion (Ridge) on the new scaled data
print("Evaluating Champion Model (Ridge)...")
ridge_model = Ridge(alpha=15.0)
ridge_cv_score = np.mean(cross_val_score(ridge_model, X_scaled, y_train_log, cv=kf, scoring='neg_root_mean_squared_error'))
print(f"Ridge (alpha=15) 10-Fold CV RMSLE: {-ridge_cv_score:.5f}")


# Model 2: The Challenger (Tuned XGBoost) on new unscaled data
print("\nTuning Challenger Model (XGBoost) (This may take a few minutes)...")
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
# Focused grid search based on common winning parameters
param_grid_xgb = {
    'n_estimators': [1000, 1500],
    'learning_rate': [0.01, 0.05],
    'max_depth': [3, 4],
    'subsample': [0.7],
    'colsample_bytree': [0.7],
    'reg_alpha': [0.005] # L1 Regularization
}
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=kf,
                               scoring='neg_root_mean_squared_error',
                               verbose=1, return_train_score=True)
grid_search_xgb.fit(X, y_train_log)
best_xgb_model = grid_search_xgb.best_estimator_
xgb_cv_score = -grid_search_xgb.best_score_
print(f"\nTuned XGBoost 10-Fold CV RMSLE: {xgb_cv_score:.5f}")
print(f"Best XGBoost Params: {grid_search_xgb.best_params_}")


# --- PHASE 3: FINAL PREDICTIONS ---
print("\n--- Phase 3: Generating Final Submissions ---")

# Train final models on ALL data
ridge_model.fit(X_scaled, y_train_log)
best_xgb_model.fit(X, y_train_log)

# --- Submission 1: Ridge Model (V2) ---
ridge_preds_log = ridge_model.predict(X_test_scaled)
ridge_preds = np.expm1(ridge_preds_log)
submission_ridge = pd.DataFrame({'Id': test_ids, 'HotelValue': ridge_preds})
submission_ridge.to_csv('submission_ridge_v2.csv', index=False)
print("Saved 'submission_ridge_v2.csv' (Tuned Ridge on new features)")

# --- Submission 2: XGBoost Model (V2) ---
xgb_preds_log = best_xgb_model.predict(X_test)
xgb_preds = np.expm1(xgb_preds_log)
submission_xgb = pd.DataFrame({'Id': test_ids, 'HotelValue': xgb_preds})
submission_xgb.to_csv('submission_xgb_v2.csv', index=False)
print("Saved 'submission_xgb_v2.csv' (Tuned XGBoost on new features)")

# --- Submission 3: Ensemble (V2) ---
# Give more weight to the model with the better CV score
if xgb_cv_score < -ridge_cv_score:
    print("XGBoost performed better, giving it 70% weight in ensemble.")
    ensemble_preds = (0.3 * ridge_preds) + (0.7 * xgb_preds)
else:
    print("Ridge performed better, giving it 70% weight in ensemble.")
    ensemble_preds = (0.7 * ridge_preds) + (0.3 * xgb_preds)
    
submission_ensemble = pd.DataFrame({'Id': test_ids, 'HotelValue': ensemble_preds})
submission_ensemble.to_csv('submission_ensemble_v2.csv', index=False)
print("Saved 'submission_ensemble_v2.csv' (Ensemble of Ridge and XGBoost)")

print("\nScript finished.")
