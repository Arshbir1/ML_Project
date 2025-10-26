# --- PHASE 1: SETUP AND LIBRARIES ---
import pandas as pd
import numpy as np
import warnings
from scipy.stats import skew
from scipy.optimize import minimize

# Models
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold # <--- THIS IS THE FIX

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

# --- PHASE 3: AGGRESSIVE PREPROCESSING ---
print("🚀 Starting aggressive preprocessing...")

# Remove outliers
df_train = df_train.drop(df_train[(df_train['UsableArea'] > 4500) & (df_train['HotelValue'] < 300000)].index)

# Log-transform the target variable
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])
y_train = df_train['HotelValue']
df_train = df_train.drop('HotelValue', axis=1)

# Combine train and test data
all_data = pd.concat((df_train, df_test)).reset_index(drop=True)

# Use the most successful feature engineering set
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['PropertyAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['OverallGrade'] = all_data['OverallQuality'] * all_data['OverallCondition']
for col in ['OverallQuality', 'UsableArea', 'TotalSF', 'PropertyAge']:
    all_data[f'{col}-s2'] = all_data[col] ** 2

# Robust Imputation
numeric_cols = all_data.select_dtypes(include=np.number).columns
all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].median())
categorical_cols = all_data.select_dtypes(include='object').columns
all_data[categorical_cols] = all_data[categorical_cols].fillna('None')

# Log-transform skewed numerical features
numeric_feats = all_data.select_dtypes(include=np.number).columns
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats = skewed_feats[skewed_feats > 0.5].index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])

# One-Hot Encode
all_data = pd.get_dummies(all_data, drop_first=True)

# Separate back into training and test sets
X = all_data[:len(y_train)].reset_index(drop=True)
X_test = all_data[len(y_train):].reset_index(drop=True)
y_train = y_train.reset_index(drop=True)

print("✅ Preprocessing complete.")

# --- PHASE 4: K-FOLD TRAINING & OOF PREDICTION ---
print("💪 Training 10 folds for each of our 3 linear models...")

kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Out-of-fold (OOF) predictions for the training set
oof_preds_lasso = np.zeros(X.shape[0])
oof_preds_ridge = np.zeros(X.shape[0])
oof_preds_elastic = np.zeros(X.shape[0])

# Predictions for the test set
test_preds_lasso = np.zeros(X_test.shape[0])
test_preds_ridge = np.zeros(X_test.shape[0])
test_preds_elastic = np.zeros(X_test.shape[0])

for fold, (train_index, val_index) in enumerate(kf.split(X, y_train)):
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

    # Lasso
    lasso = LassoCV(alphas=[0.0001, 0.0003, 0.0005], cv=5, n_jobs=-1, random_state=42).fit(X_train, y_train_fold)
    oof_preds_lasso[val_index] = lasso.predict(X_val)
    test_preds_lasso += lasso.predict(X_test) / kf.n_splits

    # Ridge
    ridge = RidgeCV(alphas=[10, 14, 18], cv=5).fit(X_train, y_train_fold)
    oof_preds_ridge[val_index] = ridge.predict(X_val)
    test_preds_ridge += ridge.predict(X_test) / kf.n_splits

    # ElasticNet
    elastic = ElasticNetCV(alphas=[0.0001, 0.0005], l1_ratio=[0.9, 0.99], cv=5, n_jobs=-1, random_state=42).fit(X_train, y_train_fold)
    oof_preds_elastic[val_index] = elastic.predict(X_val)
    test_preds_elastic += elastic.predict(X_test) / kf.n_splits
    
    print(f"  - Fold {fold+1}/10 complete.")


# --- PHASE 5: FINDING OPTIMAL BLEND WEIGHTS ---
print("🧠 Optimizing blend weights...")

# We stack our OOF predictions
stacked_oof_preds = np.vstack([oof_preds_lasso, oof_preds_ridge, oof_preds_elastic]).T

def get_rmse(weights):
    final_prediction = np.dot(stacked_oof_preds, weights)
    return np.sqrt(mean_squared_error(y_train, final_prediction))

# Initial weights (equal) and constraints (sum to 1)
initial_weights = [1/3, 1/3, 1/3]
constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = [(0, 1) for _ in range(3)]

# Find the optimal weights
res = minimize(get_rmse, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = res.x

print(f"✅ Optimal Weights Found:")
print(f"  - Lasso: {optimal_weights[0]*100:.2f}%")
print(f"  - Ridge: {optimal_weights[1]*100:.2f}%")
print(f"  - ElasticNet: {optimal_weights[2]*100:.2f}%")

# --- PHASE 6: FINAL SUBMISSION ---
print("🏆 Creating final submission with optimal blend...")

# Stack the final test predictions
stacked_test_preds = np.vstack([test_preds_lasso, test_preds_ridge, test_preds_elastic]).T

# Apply the optimal weights
blended_log_preds = np.dot(stacked_test_preds, optimal_weights)
final_predictions = np.expm1(blended_log_preds)

# Create submission file
submission = pd.DataFrame({'Id': test_ids, 'HotelValue': final_predictions})
submission.to_csv('submission_optimized_blend.csv', index=False)

print("\n🚀 Submission file 'submission_optimized_blend.csv' created successfully!")
print("This is the final, most optimized version of the linear model strategy. Good luck.")
print(submission.head())
