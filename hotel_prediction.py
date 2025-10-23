# --- PHASE 1: SETUP AND EXPLORATION ---

# 1.1 Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# 1.2 Load Data
try:
    df_train = pd.read_csv('train.csv')
    df_test = pd.read_csv('test.csv')
    print("Training and test datasets loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure train.csv and test.csv are in the same directory as the script.")
    exit()

# Store IDs for the final submission file
train_ids = df_train['Id']
test_ids = df_test['Id']
# Drop Id columns as they are not features for the model
df_train = df_train.drop('Id', axis=1)
df_test = df_test.drop('Id', axis=1)

print(f"Training data shape: {df_train.shape}")
print(f"Test data shape: {df_test.shape}")


# --- PHASE 2: PREPROCESSING AND FEATURE ENGINEERING ---

# 2.1 Log Transform Target Variable (Essential for linear models)
df_train['HotelValue'] = np.log1p(df_train['HotelValue'])

# 2.2 Combine train and test data for consistent preprocessing
# We separate the target variable first
y_train = df_train.pop('HotelValue')
all_data = pd.concat([df_train, df_test], ignore_index=True)
print(f"Combined data shape: {all_data.shape}")


# 2.3 Handle Missing Values
# Columns where NaN likely means 'None' or 'No Feature'
for col in ['PoolQuality', 'ExtraFacility', 'FacadeType', 'BoundaryFence', 'LoungeQuality',
            'ParkingType', 'ParkingFinish', 'ParkingQuality', 'ParkingCondition',
            'BasementHeight', 'BasementCondition', 'BasementExposure',
            'BasementFacilityType1', 'BasementFacilityType2']:
    all_data[col] = all_data[col].fillna('None')

# Impute other missing values using median (for numerical) or mode (for categorical)
for col in all_data.columns:
    if all_data[col].dtype == "object":
        all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    else:
        all_data[col] = all_data[col].fillna(all_data[col].median())

print(f"\nMissing values remaining after cleaning: {all_data.isnull().sum().sum()}")


# 2.4 Feature Engineering
all_data['HotelAge'] = all_data['YearSold'] - all_data['ConstructionYear']
all_data['YearsSinceRenovation'] = all_data['YearSold'] - all_data['RenovationYear']
all_data['TotalSF'] = all_data['BasementTotalSF'] + all_data['GroundFloorArea'] + all_data['UpperFloorArea']
all_data['TotalBathrooms'] = (all_data['FullBaths'] + (0.5 * all_data['HalfBaths']) +
                               all_data['BasementFullBaths'] + (0.5 * all_data['BasementHalfBaths']))

# 2.5 Encode Categorical Features
# This converts all text columns to numerical format
all_data = pd.get_dummies(all_data, drop_first=True)

# 2.6 Separate back into training and testing sets
X = all_data[:len(df_train)]
X_test = all_data[len(df_train):]

print(f"\nFinal training features shape: {X.shape}")
print(f"Final test features shape: {X_test.shape}")


# 2.7 Scale Numerical Features
# This is important for regularization models like Ridge and Lasso
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test = scaler.transform(X_test)


# --- PHASE 3: MODEL TRAINING & EXPERIMENTS ---

print("\n--- Running Model Experiments with Cross-Validation ---")

# Experiment 1: Ridge Regression
# We'll find the best alpha for Ridge. This is a common and powerful model for this type of problem.
best_ridge_score = -float('inf')
best_ridge_alpha = 0
alphas_ridge = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]

for alpha in alphas_ridge:
    ridge = Ridge(alpha=alpha)
    # Using cross-validation to get a robust score
    # Note: R^2 is used here, higher is better.
    cv_score = np.mean(cross_val_score(ridge, X, y_train, cv=10, scoring='r2'))
    print(f"Ridge (alpha={alpha}) - 10-Fold CV R^2 Score: {cv_score:.4f}")
    if cv_score > best_ridge_score:
        best_ridge_score = cv_score
        best_ridge_alpha = alpha

print(f"\nBest Ridge Alpha found: {best_ridge_alpha} with R^2 Score: {best_ridge_score:.4f}")


# Experiment 2: Lasso Regression
best_lasso_score = -float('inf')
best_lasso_alpha = 0
alphas_lasso = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]

for alpha in alphas_lasso:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    cv_score = np.mean(cross_val_score(lasso, X, y_train, cv=10, scoring='r2'))
    print(f"Lasso (alpha={alpha}) - 10-Fold CV R^2 Score: {cv_score:.4f}")
    if cv_score > best_lasso_score:
        best_lasso_score = cv_score
        best_lasso_alpha = alpha

print(f"\nBest Lasso Alpha found: {best_lasso_alpha} with R^2 Score: {best_lasso_score:.4f}")


# --- PHASE 4: FINAL PREDICTION AND SUBMISSION ---

print("\n--- Generating Submission File ---")

# Choose the best performing model based on CV scores. Let's assume it's Ridge.
final_model = Ridge(alpha=best_ridge_alpha)

# Train the final model on the ENTIRE training dataset
final_model.fit(X, y_train)

# Make predictions on the preprocessed test data
predictions_log = final_model.predict(X_test)

# Inverse transform the predictions to get the actual dollar values
predictions = np.expm1(predictions_log)

# Create the submission DataFrame
submission = pd.DataFrame({'Id': test_ids, 'HotelValue': predictions})

# Save the submission file
submission.to_csv('submission.csv', index=False)

print("\n'submission.csv' has been created successfully!")
print("You are now ready to submit to Kaggle.")