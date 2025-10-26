# --- PHASE 1: SETUP AND LIBRARIES ---
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- PHASE 2: DATA LOADING AND PREPARATION ---
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("✅ Datasets loaded successfully.")
except FileNotFoundError:
    print("❌ Error: train.csv or test.csv not found.")
    exit()

# Store Test IDs for submission
test_ids = df_test['Id']
df_train = df_train.drop('Id', axis=1)
df_test = df_test.drop('Id', axis=1)

# --- PHASE 3: AGGRESSIVE PREPROCESSING FOR TREES ---
print("🚀 Starting aggressive preprocessing tailored for LightGBM...")

# Remove outliers
df_train = df_train.drop(df_train[(df_train['UsableArea'] > 4500) & (df_train['HotelValue'] < 300000)].index)

# Log-transform the target variable
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])
y_train = df_train['HotelValue']
df_train = df_train.drop('HotelValue', axis=1)

# Combine train and test data
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

# --- FEATURE ENGINEERING ---
# Create only the most impactful features. Let the model do the rest.
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBath'] = all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) + all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths'])
all_data['PropertyAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['YearsSinceRemod'] = all_data['YearSold'] - all_data['RenovationYear']
all_data['OverallGrade'] = all_data['OverallQuality'] * all_data['OverallCondition']
all_data['TotalRooms'] = all_data['GuestRooms'] + all_data['Kitchens']
# --- END OF FEATURE ENGINEERING ---

# Impute missing values with a simple global median/mode
for col in all_data.select_dtypes(include='object').columns:
    all_data[col] = all_data[col].fillna('None')
for col in all_data.select_dtypes(include=['int64', 'float64']).columns:
    all_data[col] = all_data[col].fillna(all_data[col].median())

# *** KEY CHANGE: Convert object columns to pandas 'category' dtype for LightGBM ***
# This is far more effective than one-hot encoding for tree models.
for col in all_data.select_dtypes(include='object').columns:
    all_data[col] = all_data[col].astype('category')

# Separate back into training and test sets
X = all_data[:len(y_train)]
X_test = all_data[len(y_train):]

print("✅ Preprocessing complete.")

# --- PHASE 4: K-FOLD TRAINING OF LIGHTGBM ---
print("💪 Training 10 LightGBM models with cross-validation...")

# Define KFold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Store predictions
oof_predictions = np.zeros(X.shape[0]) # For CV score calculation
test_predictions = np.zeros(X_test.shape[0]) # For final submission

# LightGBM parameters - these are heavily tuned for this type of problem
lgb_params = {
    'objective': 'regression_l1', # L1 regression loss (robust to outliers)
    'metric': 'rmse', # Root Mean Squared Error
    'n_estimators': 2000, # Max number of trees, early stopping will optimize
    'learning_rate': 0.01, # Small learning rate requires more trees
    'feature_fraction': 0.8, # Randomly select 80% of features for each tree
    'bagging_fraction': 0.8, # Randomly select 80% of data for each tree (without resampling)
    'bagging_freq': 1, # Perform bagging on every iteration
    'lambda_l1': 0.1, # L1 regularization
    'lambda_l2': 0.1, # L2 regularization
    'num_leaves': 31, # Max number of leaves in one tree (controls complexity)
    'verbose': -1, # Suppress verbose output
    'n_jobs': -1, # Use all available CPU cores
    'seed': 42, # Random seed for reproducibility
    'boosting_type': 'gbdt', # Gradient Boosting Decision Tree
}

for fold, (train_index, val_index) in enumerate(kf.split(X, y_train)):
    print(f"--- Fold {fold+1}/10 ---")
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train_fold,
              eval_set=[(X_val, y_val_fold)], # Data to evaluate on
              eval_metric='rmse', # Metric for evaluation
              callbacks=[lgb.early_stopping(100, verbose=False)]) # Stop if score doesn't improve for 100 rounds

    # Store Out-of-Fold predictions (for evaluating CV score)
    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds

    # Add the predictions for the test set from this fold
    test_predictions += model.predict(X_test)

# Average the predictions from all 10 folds for the test set
test_predictions /= kf.n_splits

# Calculate overall Out-of-Fold RMSE (RMSLE because y_train is log-transformed)
oof_rmse = np.sqrt(mean_squared_error(y_train, oof_predictions))
print(f"\n✅ Overall Out-of-Fold RMSLE (estimate): {oof_rmse:.5f}")


# --- PHASE 5: SUBMISSION ---
print("🏆 Creating final submission file...")

# Reverse the log transformation
final_predictions = np.expm1(test_predictions)

# Create the submission file
submission = pd.DataFrame({'Id': test_ids, 'HotelValue': final_predictions})
submission.to_csv('submission_lgbm_kfold_average.csv', index=False)

print("\n🚀 Submission file 'submission_lgbm_kfold_average.csv' created successfully!")
print("This represents the LightGBM K-Fold averaging strategy.")
print(submission.head())
